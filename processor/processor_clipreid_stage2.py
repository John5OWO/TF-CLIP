import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.iotools import save_checkpoint
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from loss.softmax_loss import CrossEntropyLabelSmooth


@torch.no_grad()
def build_single_prototype_features(labels, features):
    import collections
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[int(label)].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers


@torch.no_grad()
def build_multi_prototype_features(
        features,
        labels,
        num_classes,
        num_prototypes=2,
        cluster_mode="farthest",
        normalize_prototypes=True,
        min_samples_per_proto=1,
        return_stats=False):
    if cluster_mode not in ("farthest", "duplicate_mean"):
        raise NotImplementedError("Unsupported multi-prototype cluster_mode='{}'.".format(cluster_mode))
    if cluster_mode == "farthest" and num_prototypes != 2:
        raise NotImplementedError("The farthest multi-prototype version only supports K=2.")

    labels_tensor = torch.as_tensor(labels, device=features.device, dtype=torch.long)
    prototypes = []
    sample_counts = []
    proto_cosines = []
    fallback_ids = 0
    empty_cluster_fallbacks = 0
    eps = 1e-12

    for class_idx in range(num_classes):
        class_features = features[labels_tensor == class_idx]
        sample_count = int(class_features.size(0))
        sample_counts.append(sample_count)

        if sample_count == 0:
            raise ValueError("No features found for class {} while building memory.".format(class_idx))

        mean_proto = class_features.mean(dim=0)
        if cluster_mode == "duplicate_mean":
            class_prototypes = mean_proto.unsqueeze(0).repeat(num_prototypes, 1)
        elif sample_count < num_prototypes:
            fallback_ids += 1
            class_prototypes = mean_proto.unsqueeze(0).repeat(num_prototypes, 1)
        else:
            norm_features = torch.nn.functional.normalize(class_features.float(), dim=1, eps=eps)
            norm_mean = torch.nn.functional.normalize(mean_proto.float().unsqueeze(0), dim=1, eps=eps).squeeze(0)

            dist_to_mean = 1.0 - torch.matmul(norm_features, norm_mean)
            seed_a_idx = int(torch.argmax(dist_to_mean).item())
            seed_a = class_features[seed_a_idx]
            norm_seed_a = torch.nn.functional.normalize(seed_a.float().unsqueeze(0), dim=1, eps=eps).squeeze(0)

            dist_to_seed_a = 1.0 - torch.matmul(norm_features, norm_seed_a)
            seed_b_idx = int(torch.argmax(dist_to_seed_a).item())
            seed_b = class_features[seed_b_idx]

            seeds = torch.stack([seed_a, seed_b], dim=0)
            norm_seeds = torch.nn.functional.normalize(seeds.float(), dim=1, eps=eps)
            sim_to_seeds = torch.matmul(norm_features, norm_seeds.t())
            assignments = torch.argmax(sim_to_seeds, dim=1)

            proto_list = []
            for proto_idx in range(num_prototypes):
                member_features = class_features[assignments == proto_idx]
                if int(member_features.size(0)) < min_samples_per_proto:
                    empty_cluster_fallbacks += 1
                    proto_list.append(mean_proto)
                else:
                    proto_list.append(member_features.mean(dim=0))
            class_prototypes = torch.stack(proto_list, dim=0)

        if normalize_prototypes and cluster_mode != "duplicate_mean":
            class_prototypes = torch.nn.functional.normalize(class_prototypes.float(), dim=1, eps=eps).to(features.dtype)

        if class_prototypes.size(0) == 2:
            proto_cos = torch.nn.functional.cosine_similarity(
                class_prototypes[0].float().unsqueeze(0),
                class_prototypes[1].float().unsqueeze(0),
                dim=1,
                eps=eps,
            )[0]
            proto_cosines.append(proto_cos)

        prototypes.append(class_prototypes)

    prototypes = torch.stack(prototypes, dim=0)
    if not torch.isfinite(prototypes).all():
        raise ValueError("Non-finite values found in multi-prototype memory.")

    if not return_stats:
        return prototypes

    sample_counts_tensor = torch.tensor(sample_counts, dtype=torch.float32)
    proto_cosines_tensor = torch.stack(proto_cosines).float() if proto_cosines else torch.tensor([0.0])
    stats = {
        "num_classes": num_classes,
        "num_prototypes": num_prototypes,
        "mean_samples_per_id": sample_counts_tensor.mean().item(),
        "min_samples_per_id": int(sample_counts_tensor.min().item()),
        "max_samples_per_id": int(sample_counts_tensor.max().item()),
        "fallback_ids": fallback_ids,
        "empty_cluster_fallbacks": empty_cluster_fallbacks,
        "proto_cosine_mean": proto_cosines_tensor.mean().item(),
        "proto_cosine_std": proto_cosines_tensor.std(unbiased=False).item(),
        "cluster_mode": cluster_mode,
    }
    return prototypes, stats


def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage1,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,num_classes):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_id1 = AverageMeter()
    acc_meter_id2 = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent_frame = CrossEntropyLabelSmooth(num_classes=num_classes)

    @torch.no_grad()
    def generate_cluster_features(labels, features):
        if not cfg.MODEL.MEMORY.MULTI_ENABLED:
            return build_single_prototype_features(labels, features)

        centers, stats = build_multi_prototype_features(
            features,
            labels,
            num_classes,
            num_prototypes=cfg.MODEL.MEMORY.NUM_PROTOTYPES,
            cluster_mode=cfg.MODEL.MEMORY.CLUSTER_MODE,
            normalize_prototypes=cfg.MODEL.MEMORY.NORMALIZE_PROTOTYPES,
            min_samples_per_proto=cfg.MODEL.MEMORY.MIN_SAMPLES_PER_PROTO,
            return_stats=True,
        )
        _write_multiproto_stats(stats)
        return centers

    def _write_multiproto_stats(stats):
        stats["agg_mode"] = cfg.MODEL.MEMORY.AGG_MODE
        logger.info(
            "Multi-prototype Memory Stats - num_classes {}, num_prototypes {}, "
            "mean_samples_per_id {:.4f}, min_samples_per_id {}, max_samples_per_id {}, "
            "fallback_ids {}, empty_cluster_fallbacks {}, proto_cosine_mean {:.4f}, "
            "proto_cosine_std {:.4f}, cluster_mode {}, agg_mode {}".format(
                stats["num_classes"],
                stats["num_prototypes"],
                stats["mean_samples_per_id"],
                stats["min_samples_per_id"],
                stats["max_samples_per_id"],
                stats["fallback_ids"],
                stats["empty_cluster_fallbacks"],
                stats["proto_cosine_mean"],
                stats["proto_cosine_std"],
                stats["cluster_mode"],
                stats["agg_mode"],
            )
        )

        stats_path = os.path.join(cfg.OUTPUT_DIR, "multiproto_memory_stats.txt")
        with open(stats_path, "w") as stats_file:
            stats_file.write(
                "num_classes,num_prototypes,mean_samples_per_id,min_samples_per_id,"
                "max_samples_per_id,fallback_ids,empty_cluster_fallbacks,"
                "proto_cosine_mean,proto_cosine_std,cluster_mode,agg_mode\n"
            )
            stats_file.write(
                "{},{},{:.6f},{},{},{},{},{:.6f},{:.6f},{},{}\n".format(
                    stats["num_classes"],
                    stats["num_prototypes"],
                    stats["mean_samples_per_id"],
                    stats["min_samples_per_id"],
                    stats["max_samples_per_id"],
                    stats["fallback_ids"],
                    stats["empty_cluster_fallbacks"],
                    stats["proto_cosine_mean"],
                    stats["proto_cosine_std"],
                    stats["cluster_mode"],
                    stats["agg_mode"],
                )
            )

    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    def _new_qata_stats():
        return {"img": [], "proj": []}

    @torch.no_grad()
    def _compute_qata_stats(weights):
        weights = weights.detach().float()
        eps = 1e-12
        safe_weights = weights.clamp_min(eps)
        entropy = -(safe_weights * safe_weights.log()).sum(dim=1)
        sorted_weights = torch.sort(weights, dim=1, descending=True).values
        top1 = sorted_weights[:, 0]
        if sorted_weights.size(1) >= 2:
            top2 = sorted_weights[:, :2].sum(dim=1)
        else:
            top2 = top1

        return {
            "mean": weights.mean().item(),
            "std": weights.std(unbiased=False).item(),
            "max": weights.max().item(),
            "min": weights.min().item(),
            "entropy": entropy.mean().item(),
            "effective_frames": torch.exp(entropy).mean().item(),
            "top1": top1.mean().item(),
            "top2": top2.mean().item(),
        }

    def _update_qata_stats(epoch_stats):
        if not (cfg.MODEL.QATA.ENABLED and cfg.MODEL.QATA.LOG_STATS):
            return
        latest_weights = getattr(_unwrap_model(model), "latest_qata_weights", None)
        if not latest_weights:
            return
        for name in ("img", "proj"):
            weights = latest_weights.get(name)
            if weights is not None:
                epoch_stats[name].append(_compute_qata_stats(weights))

    def _average_qata_stats(stats_list):
        if not stats_list:
            return None
        keys = stats_list[0].keys()
        return {key: sum(stats[key] for stats in stats_list) / len(stats_list) for key in keys}

    def _format_qata_stats(epoch, name, stats):
        return (
            "QATA Stats - Epoch {} {}: mean {:.4f}, std {:.4f}, max {:.4f}, min {:.4f}, "
            "entropy {:.4f}, effective_frames {:.4f}, top1 {:.4f}, top2 {:.4f}"
        ).format(
            epoch,
            name,
            stats["mean"],
            stats["std"],
            stats["max"],
            stats["min"],
            stats["entropy"],
            stats["effective_frames"],
            stats["top1"],
            stats["top2"],
        )

    def _write_qata_stats(epoch, epoch_stats):
        if not (cfg.MODEL.QATA.ENABLED and cfg.MODEL.QATA.LOG_STATS):
            return
        stats_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.QATA.STATS_FILE)
        file_exists = os.path.exists(stats_path)
        with open(stats_path, "a") as stats_file:
            if not file_exists:
                stats_file.write("epoch,branch,mean,std,max,min,entropy,effective_frames,top1,top2\n")
            for name in ("img", "proj"):
                stats = _average_qata_stats(epoch_stats[name])
                if stats is None:
                    continue
                logger.info(_format_qata_stats(epoch, name, stats))
                stats_file.write(
                    "{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                        epoch,
                        name,
                        stats["mean"],
                        stats["std"],
                        stats["max"],
                        stats["min"],
                        stats["entropy"],
                        stats["effective_frames"],
                        stats["top1"],
                        stats["top2"],
                    )
                )

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    #######   1.CLIP-Memory module ####################
    print("=> Automatically generating CLIP-Memory (might take a while, have a coffe)")
    image_features = []
    labels = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
            target = vid.to(device)  # torch.Size([64])
            if len(img.size()) == 6:
                # method = 'dense'
                b, n, s, c, h, w = img.size()
                assert (b == 1)
                img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])
                with amp.autocast(enabled=True):
                    image_feature = model(img, get_image = True)
                    image_feature = image_feature.view(-1, image_feature.size(1))
                    image_feature = torch.mean(image_feature, 0, keepdim=True)  # 1,512
                    for i, img_feat in zip(target, image_feature):
                        labels.append(i)
                        image_features.append(img_feat.cpu())
            else:
                with amp.autocast(enabled=True):
                    image_feature = model(img, get_image = True)
                    for i, img_feat in zip(target, image_feature):
                        labels.append(i)
                        image_features.append(img_feat.cpu())

        labels_list = torch.stack(labels, dim=0).cuda()  # N torch.Size([8256])
        image_features_list = torch.stack(image_features, dim=0).cuda()  # torch.Size([8256, 512])

    cluster_features = generate_cluster_features(labels_list.cpu().numpy(), image_features_list).detach()
    best_performance = 0.0
    best_epoch = 1
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        acc_meter_id1.reset()
        acc_meter_id2.reset()
        evaluator.reset()
        qata_epoch_stats = _new_qata_stats()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            with amp.autocast(enabled=True):
                B, T, C, H, W = img.shape  # B=64, T=4.C=3 H=256,W=128
                score, feat, logits1 = model(x = img, cam_label=target_cam, view_label=target_view, text_features2=cluster_features)
                _update_qata_stats(qata_epoch_stats)
                score1 = score[0:3]
                score2 = score[3]

                if (n_iter + 1) % log_period == 0:
                    loss1 = loss_fn(score1, feat, target, target_cam, logits1, isprint=True)
                else:
                    loss1 = loss_fn(score1, feat, target, target_cam, logits1)

                targetX = target.unsqueeze(1)  # 12,1   => [94 94 10 10 15 15 16 16 75 75 39 39]
                targetX = targetX.expand(B, T)
                # 12,8  => [ [94...94][94...94][10...10][10...10] ... [39...39] [39...39]]
                targetX = targetX.contiguous()
                targetX = targetX.view(B * T,
                                       -1)  # 96  => [94...94 10...10 15...15 16...16 75...75 39...39]
                targetX = targetX.squeeze(1)
                loss_frame = xent_frame(score2, targetX)
                loss = loss1 + loss_frame / T


            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc1 = (logits1.max(1)[1] == target).float().mean()
            acc_id1 = (score[0].max(1)[1] == target).float().mean()
            acc_id2 = (score[3].max(1)[1] == targetX).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc1, 1)
            acc_meter_id1.update(acc_id1, 1)
            acc_meter_id2.update(acc_id2, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc_clip: {:.3f}, Acc_id1: {:.3f}, Acc_id2: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader_stage2),
                            loss_meter.avg, acc_meter.avg, acc_meter_id1.avg, acc_meter_id2.avg, scheduler.get_lr()[0]))

        scheduler.step()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))
        _write_qata_stats(epoch, qata_epoch_stats)


        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10, 20]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else:
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else:
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10, 20]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
            prec1 = cmc[0] + mAP
            is_best = prec1 > best_performance
            best_performance = max(prec1, best_performance)
            if is_best:
                best_epoch = epoch
            save_checkpoint(model.state_dict(), is_best, os.path.join(cfg.OUTPUT_DIR, 'checkpoint_ep.pth.tar'))

    logger.info("==> Best Perform {:.1%}, achieved at epoch {}".format(best_performance, best_epoch))
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference_dense(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("TFCLIP.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        if len(img.size()) == 6:
            # method = 'dense'
            b, n, s, c, h, w = img.size()
            assert (b == 1)
            img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])

        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            feat = feat.view(-1, feat.size(1))
            feat = torch.mean(feat, 0, keepdim=True)  # 1,512
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def do_inference_rrs(cfg,
                     model,
                     val_loader,
                     num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            # model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        if len(img.size()) == 6:
            # method = 'dense'
            b, n, s, c, h, w = img.size()
            assert (b == 1)
            img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])

        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            # feat = feat.view(-1, feat.size(1))
            # feat = torch.mean(feat, 0, keepdim=True)  # 1,512
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
