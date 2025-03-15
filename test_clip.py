import torch
import clip
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cairosvg
import io
import re
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

sketch_folder = Path(r'C:\Users\sentuix\code\sc\CLIPasso\best')

svg_files = list(sketch_folder.glob('*.svg'))
png_files = list(sketch_folder.glob('*.png'))
jpg_files = list(sketch_folder.glob('*.jpg'))
jpeg_files = list(sketch_folder.glob('*.jpeg'))

image_files = svg_files + png_files + jpg_files + jpeg_files
print(f"找到 {len(svg_files)} 个SVG文件, {len(png_files)} 个PNG文件, {len(jpg_files) + len(jpeg_files)} 个JPG文件")
print(f"总共找到 {len(image_files)} 个图像文件")

class_names = ["cat", "dog", "horse", "person", "car", "airplane", "chair", "bottle", "watch", "tree", "camel", "rose", "sunflower", "elephant", "giraffe", "kangaroo", "panda", "penguin", "tiger", "zebra"]

text_prompts = [f"A sketch of a(n) {name}" for name in class_names]
text_inputs = clip.tokenize(text_prompts).to(device)

classification_results = {}
confidence_scores = {}

temp_dir = Path("temp_png")
temp_dir.mkdir(exist_ok=True)

for img_path in image_files:
    try:
        png_path = temp_dir / f"{img_path.stem}.png"
        
        if img_path.suffix.lower() == '.svg':
            try:
                png_data = cairosvg.svg2png(url=str(img_path), background_color="white")
                image = Image.open(io.BytesIO(png_data))
            except Exception as svg_error:
                print(f"SVG转换错误 {img_path}: {svg_error}")
                try:
                    png_data = cairosvg.svg2png(url=str(img_path))
                    svg_image = Image.open(io.BytesIO(png_data))
                    
                    white_bg = Image.new("RGB", svg_image.size, (255, 255, 255))
                    
                    if svg_image.mode == 'RGBA':
                        white_bg.paste(svg_image, (0, 0), svg_image.split()[3])
                    else:
                        white_bg.paste(svg_image, (0, 0))
                    
                    image = white_bg
                except Exception as e:
                    print(f"无法处理SVG文件 {img_path}: {e}")
                    continue
        else:
            image = Image.open(img_path)
            
        if image.mode == 'RGBA':
            white_bg = Image.new("RGB", image.size, (255, 255, 255))
            white_bg.paste(image, (0, 0), image.split()[3])
            image = white_bg
        
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            values, indices = similarity[0].topk(3)
            
            top_class = class_names[indices[0].item()]
            classification_results[img_path.name] = top_class
            confidence_scores[img_path.name] = {
                class_names[indices[i].item()]: values[i].item() for i in range(3)
            }
            
            print(f"图像: {img_path.name} | 预测类别: {top_class} | 置信度: {values[0].item():.2f}")
            print(f"  次要预测: {class_names[indices[1].item()]} ({values[1].item():.2f}), {class_names[indices[2].item()]} ({values[2].item():.2f})")
            
    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {e}")

for temp_file in temp_dir.glob('*'):
    try:
        os.remove(temp_file)
    except Exception as e:
        print(f"无法删除临时文件 {temp_file}: {e}")

try:
    os.rmdir(temp_dir)
except Exception as e:
    print(f"无法删除临时目录 {temp_dir}: {e}")

class_counts = {}
for class_name in classification_results.values():
    if class_name in class_counts:
        class_counts[class_name] += 1
    else:
        class_counts[class_name] = 1

print("\n分类统计结果:")
for class_name, count in class_counts.items():
    print(f"类别: {class_name} | 数量: {count} | 比例: {count / len(image_files):.2%}")

import matplotlib.pyplot as plt

classes = list(class_counts.keys())
counts = list(class_counts.values())
percentages = [count / len(image_files) * 100 for count in counts]

plt.figure(figsize=(12, 6))
bars = plt.bar(classes, counts, color='skyblue')

for bar, percentage in zip(bars, percentages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height}个 ({percentage:.1f}%)',
             ha='center', va='bottom', fontsize=9)

plt.title('图像分类结果统计')
plt.xlabel('类别')
plt.ylabel('图像数量')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('classification_results.png')
plt.show()

plt.figure(figsize=(12, 10))
plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90, shadow=True, 
        labeldistance=1.1,
        pctdistance=0.85,   
        textprops={'fontsize': 10})

plt.legend(classes, loc="best", bbox_to_anchor=(0.9, 0.1, 0.5, 0.5))

plt.axis('equal')
plt.title('各类别图像占比')
plt.tight_layout()

plt.savefig('classification_pie_chart.png', dpi=300)
plt.show()

print(f"\n可视化结果已保存为 'classification_results.png' 和 'classification_pie_chart.png'")