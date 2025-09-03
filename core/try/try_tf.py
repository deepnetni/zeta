import os
from torch.utils.tensorboard.writer import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator


def rebuild_tb_log(old_logdir, new_logdir, extra_points):
    ea = event_accumulator.EventAccumulator(
        old_logdir, size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    print(ea.Tags())

    all_data = []
    for tag in ea.Tags()["scalars"]:
        for e in ea.Scalars(tag):
            all_data.append((tag, e.step, e.value))

    print(all_data)
    # for tag, pts in extra_points.items():
    #     for step, val in pts:
    #         all_data.append((tag, step, val))

    # all_data.sort(key=lambda x: x[1])

    # os.makedirs(new_logdir, exist_ok=True)
    # writer = SummaryWriter(new_logdir)
    # for tag, step, val in all_data:
    #     writer.add_scalar(tag, val, step)
    # writer.close()


if __name__ == "__main__":
    f = "~/model_results_trunk/FIG6/fig6_GAN_dns/baseline_aff/tfb_t/"

    out = os.path.dirname(__file__)
    writer = rebuild_tb_log(f, out, None)
