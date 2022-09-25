# coding:utf-8
# 将预测的目标句子与源句子比较，得到edits

import sys
import Levenshtein
import json
import string

src_path = sys.argv[1] # 不带sid的source
tgt_path = sys.argv[2] # 不带sid的target
# sid_path = sys.argv[3] # sid或带sid的source

with open(src_path) as f_src, open(tgt_path) as f_tgt:
    lines_src = f_src.readlines()
    lines_tgt = f_tgt.readlines()
    # lines_sid = f_sid.readlines()
    assert len(lines_src) == len(lines_tgt)

    for i in range(len(lines_src)):
        
        lines_src[i] = lines_src[i].split('\t')
        lines_tgt[i] = lines_tgt[i].split('\t')

        if len(lines_src[i]) == 2:
            sid = lines_src[i][0]
            src_line = ''.join(lines_src[i][1].strip().replace(',', '，').split())
            tgt_line = ''.join(lines_tgt[i][1].strip().replace(',', '，').split())
        else:
            sid = i
            src_line = ''.join(lines_src[i][0].strip().replace(',', '，').split())
            tgt_line = ''.join(lines_tgt[i][0].strip().replace(',', '，').split())

        
        # sid = lines_sid[i].strip().split('\t')[0]

        # edits = Levenshtein.opcodes(src_line, tgt_line)
        # reverse
        _edits = Levenshtein.opcodes(src_line[::-1], tgt_line[::-1])[::-1]
        edits = []
        src_len = len(src_line)
        tgt_len = len(tgt_line)
        for edit in _edits:
            edits.append((edit[0], src_len - edit[2], src_len - edit[1], tgt_len - edit[4], tgt_len - edit[3]))

        # merge coterminous Levenshtein edited spans
        merged_edits = []
        for edit in edits:
            if edit[0] == 'equal':
                continue
            if len(merged_edits) > 0:
                last_edit = merged_edits[-1]
                if last_edit[2] == edit[1]:
                    assert last_edit[4] == edit[3]
                    new_edit = ('hybrid', last_edit[1], edit[2], last_edit[3], edit[4])
                    merged_edits[-1] = new_edit
                elif last_edit[0] == 'insert' and edit[0] == 'delete' \
                    and tgt_line[last_edit[3]:last_edit[4]] == src_line[edit[1]:edit[2]]:
                    new_edit = ('luanxu', last_edit[1], edit[2], last_edit[3], edit[4])
                    merged_edits[-1] = new_edit
                elif last_edit[0] == 'delete' and edit[0] == 'insert' \
                    and src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]:edit[4]]:
                    new_edit = ('luanxu', last_edit[1], edit[2], last_edit[3], edit[4])
                    merged_edits[-1] = new_edit
                else:
                    merged_edits.append(edit)
            else:
                merged_edits.append(edit)

        # generate edit sequence
        result = []
        for edit in merged_edits:
            if tgt_line[edit[3]:edit[4]] == '[UNK]':
                continue
            if edit[0] == "insert":
                result.append((str(edit[1]+1), str(edit[1]+1), "M", tgt_line[edit[3]:edit[4]]))
            elif edit[0] == "replace":
                # new_op = post_process_S(src_line, (str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
                result.append((str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
            elif edit[0] == "delete":
                result.append((str(edit[1]+1), str(edit[2]), "R"))
            elif edit[0] == "hybrid":
                # new_op = post_process_S(src_line, (str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
                result.append((str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
            elif edit[0] == "luanxu":
                result.append((str(edit[1]+1), str(edit[2]) , "W"))

        # print
        # out_line = id +',\t'
        if result:
            for res in result:
                print(str(sid) + ',\t'+',\t'.join(res))
        else:
            print(str(sid)+',\tcorrect')
        