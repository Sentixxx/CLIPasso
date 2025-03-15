import sys
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import multiprocessing as mp
import os
import subprocess as sp
from shutil import copyfile

import numpy as np
import torch
from IPython.display import Image as Image_colab
from IPython.display import display, SVG, clear_output
from ipywidgets import IntSlider, Output, IntProgress, Button
import time
import glob
from pathlib import Path
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--target_file", type=str,
                    help="target image file, located in <target_images>")
parser.add_argument("--num_strokes", type=int, default=16,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
parser.add_argument("--num_iter", type=int, default=2001,
                    help="number of iterations")
parser.add_argument("--fix_scale", type=int, default=0,
                    help="if the target image is not squared, it is recommended to fix the scale")
parser.add_argument("--mask_object", type=int, default=0,
                    help="if the target image contains background, it's better to mask it out")
parser.add_argument("--num_sketches", type=int, default=3,
                    help="it is recommended to draw 3 sketches and automatically chose the best one")
parser.add_argument("--multiprocess", type=int, default=0,
                    help="recommended to use multiprocess if your computer has enough memory")
parser.add_argument('-colab', action='store_true')
parser.add_argument('-cpu', action='store_true')
parser.add_argument('-display', action='store_true')
parser.add_argument('--gpunum', type=int, default=0)
parser.add_argument('--eval', action='store_true', help='使用评估模式，从指定目录递归处理图片')

args = parser.parse_args()

multiprocess = not args.colab and args.num_sketches > 1 and args.multiprocess

abs_path = os.path.abspath(os.getcwd())

# 只有在非评估模式下才检查目标文件
if not args.eval:
    target = f"{abs_path}/target_images/{args.target_file}"
    assert os.path.isfile(target), f"{target} does not exists!"
else:
    target = ""

if not os.path.isfile(f"{abs_path}/U2Net_/saved_models/u2net.pth"):
    sp.run(["gdown", "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
           "-O", "U2Net_/saved_models/"])

test_name = os.path.splitext(args.target_file)[0]
output_dir = f"{abs_path}/output_sketches/{test_name}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

num_iter = args.num_iter
save_interval = 10
use_gpu = not args.cpu

if not torch.cuda.is_available():
    use_gpu = False
    print("CUDA is not configured with GPU, running with CPU instead.")
    print("Note that this will be very slow, it is recommended to use colab.")

#结果显示
if args.colab:
    print("=" * 50)
    print(f"Processing [{args.target_file}] ...")
    if args.colab or args.display:
        img_ = Image_colab(target)
        display(img_)
        print(f"GPU: {use_gpu}, {torch.cuda.current_device()}")
    print(f"Results will be saved to \n[{output_dir}] ...")
    print("=" * 50)

# seeds = list(range(0, args.num_sketches * 1000, 1000))

seeds = [0]

def run_eval(seed, wandb_name):
    source_dir = "" 
    if source_dir == "":
        print("请输入评估图片的目录!")
        return
    eval_dir = os.path.join(abs_path, "eval")
    best_dir = os.path.join(abs_path, "best")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
    
    for img_path in image_files:
        rel_path = os.path.relpath(img_path, source_dir)
        output_dirname = os.path.dirname(rel_path)
        img_basename = os.path.basename(img_path)
        img_name_without_ext = os.path.splitext(img_basename)[0]
        
        current_output_dir = os.path.join(eval_dir, output_dirname)
        result_dir = os.path.join(best_dir, output_dirname)
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir, exist_ok=True)
        
        target_size = (224, 224)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}，跳过")
            continue
            
        h, w = img.shape[:2]
        ratio = min(target_size[0] / w, target_size[1] / h)
        new_size = (int(w * ratio), int(h * ratio))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
            
        canvas = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        
        x_offset = (target_size[0] - w) // 2
        y_offset = (target_size[1] - h) // 2
        
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img
        
        temp_img_path = os.path.join(current_output_dir, f"temp_{img_basename}")
        cv2.imwrite(temp_img_path, canvas)
        
        current_wandb_name = f"{img_name_without_ext}_{args.num_strokes}strokes_seed{seed}"
        
        print(f"处理图片: {img_path}")
        print(f"规范化尺寸后保存到: {temp_img_path}")
        print(f"输出到: {current_output_dir}")
        
        if not os.path.exists(best_dir):
            os.makedirs(best_dir)
        
        exit_code = sp.run(["python", "painterly_rendering_new.py", temp_img_path,
                            "--num_paths", str(args.num_strokes),
                            "--output_dir", current_output_dir,
                            "--wandb_name", current_wandb_name,
                            "--num_iter", "1001",
                            "--save_interval", str(save_interval),
                            "--seed", str(seed),
                            "--use_gpu", str(int(use_gpu)),
                            "--fix_scale", str(args.fix_scale),
                            "--mask_object", str(args.mask_object),
                            "--mask_object_attention", str(args.mask_object),
                            "--display_logs", str(int(args.colab)),
                            "--display", str(int(args.display))])
        
        if exit_code.returncode:
            print(f"处理图片 {img_path} 时出错")
            continue
        
        try:
            config_path = os.path.join(current_output_dir, current_wandb_name, "config.npy")
            if os.path.exists(config_path):
                config = np.load(config_path, allow_pickle=True)[()]
                loss_eval = np.array(config['loss_eval'])
                inds = np.argsort(loss_eval)
                losses_all[current_wandb_name] = loss_eval[inds][0]
                
                src_svg = os.path.join(current_output_dir, current_wandb_name, "best_iter.svg")
                # src_png = os.path.join(current_output_dir, current_wandb_name, "best_iter.jpg")
                dst_svg = os.path.join(result_dir, f"{current_wandb_name}_best.svg")
                # dst_png = os.path.join(result_dir, f"{current_wandb_name}_best.jpg")
                if os.path.exists(src_svg):
                    copyfile(src_svg, dst_svg)
                # if os.path.exists(src_png):
                    # copyfile(src_png, dst_png)
            try:
                svg_logs_dir = os.path.join(current_output_dir, current_wandb_name, "svg_logs")
                if os.path.exists(svg_logs_dir):
                    for file in os.listdir(svg_logs_dir):
                        file_path = os.path.join(svg_logs_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    os.rmdir(svg_logs_dir)
                    
                temp_files = ["loss.npy", "paths.npy", "points.npy", "strokes.npy"]
                for temp_file in temp_files:
                    temp_path = os.path.join(current_output_dir, current_wandb_name, temp_file)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                print(f"已清理 {current_output_dir} 内的临时文件")
            except Exception as e:
                print(f"清理文件时出错: {e}")
        except Exception as e:
            print(f"处理配置文件时出错: {e}")

def run(seed, wandb_name):
    print(target)
    exit_code = sp.run(["python", "painterly_rendering_new.py", target,
                            "--num_paths", str(args.num_strokes),
                            "--output_dir", output_dir,
                            "--wandb_name", wandb_name,
                            "--num_iter", str(num_iter),
                            "--save_interval", str(save_interval),
                            "--seed", str(seed),
                            "--use_gpu", str(int(use_gpu)),
                            "--fix_scale", str(args.fix_scale),
                            "--mask_object", str(args.mask_object),
                            "--mask_object_attention", str(
                                args.mask_object),
                            "--display_logs", str(int(args.colab)),
                            "--display", str(int(args.display))])
    if exit_code.returncode:
        sys.exit(1)

    config = np.load(f"{output_dir}/{wandb_name}/config.npy",
                     allow_pickle=True)[()]
    #提取损失评估数据
    loss_eval = np.array(config['loss_eval'])
    inds = np.argsort(loss_eval)
    losses_all[wandb_name] = loss_eval[inds][0]
 
#展示图片
def display_(seed, wandb_name):
    path_to_svg = f"{output_dir}/{wandb_name}/svg_logs/"
    intervals_ = list(range(0, num_iter, save_interval))
    filename = f"svg_iter0.svg"
    #显示滑块和输出框
    display(IntSlider())
    out = Output()
    display(out)
    for i in intervals_:
        filename = f"svg_iter{i}.svg"
        not_exist = True 
        while not_exist:
            not_exist = not os.path.isfile(f"{path_to_svg}/{filename}")
            continue
        with out:
            clear_output()
            print("")
            display(IntProgress(
                        value=i,
                        min=0,
                        max=num_iter,
                        description='Processing:',
                        bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
                        style={'bar_color': 'maroon'},
                        orientation='horizontal'
                    ))
            display(SVG(f"{path_to_svg}/svg_iter{i}.svg"))

if __name__ == "__main__":
    mp.freeze_support()
    
    exit_codes = []
    
    if args.eval:
        # 评估模式：使用单个seed直接调用run_eval
        print("启动评估模式，从指定目录递归处理图片...")
        losses_all = {}
        seed = seeds[0]
        wandb_name = f"{test_name}_{args.num_strokes}strokes_seed{seed}"
        run_eval(seed, wandb_name)
    else:
        manager = mp.Manager()
        losses_all = manager.dict()
        
        if multiprocess:
            ncpus = 10
            P = mp.Pool(ncpus)  # Generate pool of workers
            
        for seed in seeds:
            wandb_name = f"{test_name}_{args.num_strokes}strokes_seed{seed}"
            if multiprocess:
                P.apply_async(run, (seed, wandb_name))
            else:
                run(seed, wandb_name)
    
        if args.display:
            time.sleep(10)
            if multiprocess:
                P.apply_async(display_, (0, f"{test_name}_{args.num_strokes}strokes_seed0"))
            else:
                display_(0, f"{test_name}_{args.num_strokes}strokes_seed0")
    
        if multiprocess:
            P.close()
            P.join()  # start processes
            
        sorted_final = dict(sorted(losses_all.items(), key=lambda item: item[1]))
        if sorted_final:
            copyfile(f"{output_dir}/{list(sorted_final.keys())[0]}/best_iter.svg",
                    f"{output_dir}/{list(sorted_final.keys())[0]}_best.svg")