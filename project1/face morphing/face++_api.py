import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using Face++ Detect API to get 106 keypoints of a given picture")
    parser.add_argument("--src", type=str, help="Source image", required=True)
    parser.add_argument("--target", type=str, help="Target image", required=True)
    parser.add_argument("--key", type=str, help="API key", default="UAyl_WtkOcXouoLaiH_YJ91jUkrYXJC4")
    parser.add_argument("--secret", type=str, help="API Secret", default="a8v8zXGuruM80Vbm7FOGly1RRJM5eh3t")
    args = parser.parse_args()
    os.system(
        'curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/detect" -F "api_key={}" -F "api_secret={}" -F "image_file=@{}" -F "return_landmark=2" > {}.json'.format(
            args.key, args.secret, args.src, args.src
        )
    )
    os.system(
        'curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/detect" -F "api_key={}" -F "api_secret={}" -F "image_file=@{}" -F "return_landmark=2" > {}.json'.format(
            args.key, args.secret, args.target, args.target
        )
    )
    with open(args.src + ".json", "r") as f:
        raw_src = json.load(f)
    with open(args.target + ".json", "r") as f:
        raw_target = json.load(f)
    lst_src, lst_target = [], []
    s_src, s_target = set(), set()
    for i, (val_src, val_target) in enumerate(zip(raw_src["faces"][0]["landmark"].values(), raw_target["faces"][0]["landmark"].values())):
        p_src, p_target = (val_src["y"], val_src["x"]), (val_target["y"], val_target["x"])
        if p_src not in s_src and p_target not in s_target:
            s_src.add(p_src)
            s_target.add(p_target)
            lst_src.append(p_src)
            lst_target.append(p_target)
        else:
            print("Duplicated at position", i)
    print("Number of keypoints:", len(lst_src))
    with open(args.src + ".json", "w") as f:
        json.dump(lst_src, f)
    with open(args.target + ".json", "w") as f:
        json.dump(lst_target, f)