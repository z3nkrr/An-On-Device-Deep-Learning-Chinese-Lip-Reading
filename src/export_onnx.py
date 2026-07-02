"""
export_onnx.py — 將訓練好的 FullModel checkpoint 匯出成 ONNX
（對應 SE-Res + Bi-GRU + MS-TCN 架構版本的 model.py）

用法（PowerShell）:
python export_onnx.py --ckpt checkpoints\best.pt --out checkpoints\model.onnx --seq-len 40 --spatial-size 88

說明：
- 自動從 checkpoint 推斷 num_classes（讀取 classifier.1.weight 的 shape）
- 匯出兩個輸出: 'logits' (B, num_classes) 與 'seq' (B, T', 256)
- 使用 dynamic_axes，讓匯出的 ONNX 可接受不同的 batch / time 長度
"""

import os
import sys
import argparse
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import FullModel


def infer_num_classes_from_ckpt(ckpt):
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    # FullModel 的分類層是 classifier.1.weight (nn.Sequential: Dropout, Linear)
    for k, v in sd.items():
        if k.endswith('classifier.1.weight'):
            return v.shape[0]
    # 後備：找任何以 fc.weight 或 classifier 結尾的權重
    for k, v in sd.items():
        if 'classifier' in k and k.endswith('.weight') and v.dim() == 2:
            return v.shape[0]
    raise RuntimeError('無法從 checkpoint 推斷 num_classes，請確認權重鍵名')


def load_state_dict_into_model(model, ckpt):
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    new_sd = {}
    for k, v in sd.items():
        nk = k[len('module.'):] if k.startswith('module.') else k
        new_sd[nk] = v
    model.load_state_dict(new_sd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='訓練好的權重路徑 (.pt)')
    parser.add_argument('--out',  required=True, help='輸出 ONNX 路徑')
    parser.add_argument('--seq-len',      type=int, default=40, help='time 長度（dummy input）')
    parser.add_argument('--spatial-size', type=int, default=88, help='影像 H/W（dummy input）')
    parser.add_argument('--in-channels',  type=int, default=1,  help='輸入 channel 數')
    parser.add_argument('--opset',        type=int, default=13, help='ONNX opset 版本')
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f'[✗] 找不到 checkpoint: {args.ckpt}')
        return

    ckpt = torch.load(args.ckpt, map_location='cpu')
    num_classes = infer_num_classes_from_ckpt(ckpt)
    print(f'[✓] 推斷 num_classes = {num_classes}')

    # 對應 SE-Res + Bi-GRU + MS-TCN 版本的 FullModel
    model = FullModel(num_classes=num_classes,
                      in_channels=args.in_channels,
                      dropout=0.0)  # 匯出時關掉 dropout
    load_state_dict_into_model(model, ckpt)
    model.eval()
    model.to('cpu')

    # dummy input: (B, T, C, H, W)
    dummy = torch.zeros((1, args.seq_len, args.in_channels,
                         args.spatial_size, args.spatial_size), dtype=torch.float)

    output_path = args.out
    output_dir  = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_names  = ['input']
    output_names = ['logits', 'seq']
    # 靜態 shape 匯出：不傳 dynamic_axes，batch/time 完全固定為 dummy input 的大小
    # 這樣 onnx2tf 轉 TFLite 時不需要處理動態維度，速度快很多也更穩定

    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                (dummy,),
                output_path,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamo=False,
            )
            print(f'[✓] ONNX 匯出成功: {output_path}')

            # 簡單驗證輸出 shape
            logits, seq = model(dummy)
            print(f'    logits shape : {tuple(logits.shape)}')
            print(f'    seq shape    : {tuple(seq.shape)}')

        except Exception as e:
            print(f'[✗] ONNX 匯出失敗: {repr(e)}')
            raise


if __name__ == '__main__':
    main()