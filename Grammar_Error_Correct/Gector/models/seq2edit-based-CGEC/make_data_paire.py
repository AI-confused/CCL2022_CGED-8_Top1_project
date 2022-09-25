import json

file_all = "/home/wangzhihao/jingsai/cltc/MuCGEC/data_track2/zengqiang_lang8/lang8_data_v3_aug.jsonl"

file_scr = "/home/wangzhihao/jingsai/cltc/MuCGEC/data_track2/zengqiang_lang8/lang8_data_src.txt"
file_trg = "/home/wangzhihao/jingsai/cltc/MuCGEC/data_track2/zengqiang_lang8/lang8_data_trg.txt"

with open(file_scr, "w") as fs:
    with open(file_trg, "w") as ft:
        with open(file_all, "r") as f_all:
            for line in f_all:
                data = json.loads(line)
                src = data["text"]
                trg = data["correct"]
                fs.write(src + "\n")
                ft.write(trg + "\n")


