#!/usr/bin/env python3

import argparse
import glob
import numpy as np
import pandas as pd


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="action_stats.npz")
    args = parser.parse_args()


    files = glob.glob(
        args.data + "/**/*.parquet",
        recursive=True
    )

    print("found parquet:", len(files))


    actions = []


    for f in files:
        print("loading", f)

        df = pd.read_parquet(f)

        if "action" not in df.columns:
            print("skip no action:", f)
            continue

        for a in df["action"]:
            actions.append(
                np.asarray(a, dtype=np.float32)
            )


    actions = np.stack(actions)


    print("================")
    print("action shape:", actions.shape)
    print("================")


    mean = actions.mean(axis=0)
    std = actions.std(axis=0)

    amin = actions.min(axis=0)
    amax = actions.max(axis=0)

    p01 = np.percentile(actions,1,axis=0)
    p99 = np.percentile(actions,99,axis=0)


    for i in range(actions.shape[1]):

        print(
            f"""
dim {i}
 mean={mean[i]:.5f}
 std ={std[i]:.5f}
 min ={amin[i]:.5f}
 max ={amax[i]:.5f}
 p01 ={p01[i]:.5f}
 p99 ={p99[i]:.5f}
"""
        )


    np.savez(
        args.out,
        mean=mean,
        std=std,
        min=amin,
        max=amax,
        p01=p01,
        p99=p99,
    )

    print("saved:",args.out)



if __name__=="__main__":
    main()