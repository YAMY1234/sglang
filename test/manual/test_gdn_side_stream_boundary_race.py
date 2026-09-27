"""#884 GPU reproduction of the graph-on 1+2 prefix-checkpoint race (real CUDA streams, no model).

Sequence per completing prompt: the side stream publishes prefix_valid[slot] = 1 and copies it to the radix track
slot; the boundary DECODE token then invalidates prefix_valid[slot] on the forward stream. Without a join the
invalidation can land before the side-stream copy, so the track slot inherits 0. With FactoredChunkState-style
join()/mark() (event wait on the forward stream) the track slot is always 1.
usage: python test_gdn_side_stream_boundary_race.py [--trials N] [--out result.json]
"""
import argparse
import json

import torch


def trial(join: bool, sleep_cycles: int) -> int:
    dev = torch.device("cuda")
    prefix_valid = torch.zeros(8, dtype=torch.int32, device=dev)
    slot, track = 3, 5
    forward = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    done = torch.cuda.Event()
    side.wait_stream(forward)
    with torch.cuda.stream(side):
        torch.cuda._sleep(sleep_cycles)                   # the prefill-end commit graph's work
        prefix_valid[slot] = 1                            # publish P checkpoint of the committed slot
        prefix_valid[track] = prefix_valid[slot]          # radix final copy (copy_slots)
    done.record(side)
    if join:
        forward.wait_event(done)                          # init_forward_metadata_out_graph -> spec.join()
    prefix_valid[slot] = 0                                # boundary DECODE token invalidates the slot's P state
    torch.cuda.synchronize()
    return int(prefix_valid[track].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sleep", type=int, default=50_000_000)
    ap.add_argument("--out")
    args = ap.parse_args()
    unjoined = [trial(False, args.sleep) for _ in range(args.trials)]
    joined = [trial(True, args.sleep) for _ in range(args.trials)]
    result = dict(trials=args.trials, unjoined_track_valid=sum(unjoined), joined_track_valid=sum(joined),
                  race_reproduced=sum(unjoined) < args.trials, fix_holds=sum(joined) == args.trials)
    print(json.dumps(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    assert result["fix_holds"], "joined sequence lost the track checkpoint"


if __name__ == "__main__":
    main()
