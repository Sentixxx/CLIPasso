import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import sys
import time
import traceback

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm, trange

import matplotlib.pyplot as plt

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer
from IPython.display import display, SVG
from gcn import GCN


def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_, mask

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
color_array = []


def render_matrix(renderer,path):
    adjacency_matrix = renderer.get_adj_matrix()
    # print(adjacency_matrix)
    num_elements = adjacency_matrix.shape[0]
    
    # 创建一个2行1列的布局，第一行为邻接矩阵，第二行为颜色条
    fig, ax = plt.subplots(nrows=2, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 0.05]})
    
    # 设置第一个子图：邻接矩阵
    im = ax[0].imshow(adjacency_matrix.cpu().detach().numpy(),
                      cmap='RdYlBu_r',
                      vmin=0,
                      vmax=1,
                      interpolation='nearest')
    
    # 添加网格线
    for i in range(num_elements + 1):
        ax[0].axvline(x=i - 0.5, color='black', linewidth=1)
        ax[0].axhline(y=i - 0.5, color='black', linewidth=1)
    
    # 添加数值标注
    for i in range(num_elements):
        for j in range(num_elements):
            text_color = 'white' if adjacency_matrix[i, j] > 0.8 else 'black'
            ax[0].text(j, i, f'{adjacency_matrix[i, j]:.2f}',
                       ha='center', va='center',
                       color=text_color, fontsize=8)
    
    # 设置刻度，间隔为1
    ax[0].set_xticks(range(num_elements))
    ax[0].set_yticks(range(num_elements))
    ax[0].set_xticklabels(range(num_elements))
    ax[0].set_yticklabels(range(num_elements))
    ax[0].set_title('Similarity Matrix', fontsize=14, pad=20)
    
    # 添加外边框
    for spine in ax[0].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # 创建横向颜色条并将颜色映射存储到color_array
    norm = Normalize(vmin=0, vmax=num_elements - 1)
    cmap = cm.get_cmap('gist_rainbow')
    color_array[:] = [cmap(norm(i)) for i in range(num_elements)]  # 将颜色映射保存到color_array
    
    # 设置第二个子图：横向颜色条
    ax[1].imshow([np.linspace(0, 1, num_elements)], aspect='auto', cmap='gist_rainbow')
    # ax[1].set_axis_off()  # 不显示坐标轴
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax[0], orientation='horizontal', pad=0.1)
    cbar.set_label('Similarity', fontsize=12)
    
    # 显示图形
    plt.tight_layout()
    # plt.show()
    plt.savefig(path)


import pydiffvg

def render_colored_img(render):
    control_points_set = []
    for points in render.control_points_set:
        control_points_set.append(points.clone())
    shapes = []
    shape_groups = []
    for i , points in enumerate(control_points_set):
        color = torch.tensor(color_array[i])
        path =  pydiffvg.Path(num_control_points = render.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(render.width),
                                is_closed = False)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, stroke_color= color)
        shape_groups.append(path_group)

    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        render.canvas_width, render.canvas_height, shapes, shape_groups)
    img = _render(render.canvas_width, # width
                render.canvas_height, # height
                2,   # num_samples_x
                2,   # num_samples_y
                0,   # seed
                None,
                *scene_args)
    opacity = img[:, :, 3:4]
    img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = 'cuda') * (1 - opacity)
    img = img[:, :, :3]
    # Convert img from HWC to NCHW
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2).to('cuda') # NHWC -> NCHW
    return img

def render_img(control_points_set, render):
    # print(control_points_set)
    control_points_set = control_points_set * render.canvas_width
    render.shapess = []
    render.shape_groupss = []
    for i , points in enumerate(control_points_set):
        color = torch.tensor([0.0, 0.0, 0.0, 1.0])
        path =  pydiffvg.Path(num_control_points = render.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(render.width),
                                is_closed = False)
        render.shapess.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(render.shapess) - 1]), fill_color = None, stroke_color= color)
        render.shape_groupss.append(path_group)

    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        render.canvas_width, render.canvas_height, render.shapess, render.shape_groupss)
    img = _render(render.canvas_width, # width
                render.canvas_height, # height
                2,   # num_samples_x
                2,   # num_samples_y
                0,   # seed
                None,
                *scene_args)
    opacity = img[:, :, 3:4]
    img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = 'cuda') * (1 - opacity)
    img = img[:, :, :3]
    # Convert img from HWC to NCHW
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2).to('cuda') # NHWC -> NCHW
    return img


import networkx as nx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision.utils import make_grid

def render_graph(features_matrix, adj_matrix, path):
    # 获取邻接矩阵和特征矩阵
    adj_matrix = adj_matrix.clone().detach().cpu().numpy()  # 已经在GPU上转移过来的tensor，直接转为numpy
    features_matrix = features_matrix.clone().detach().cpu().numpy()
    
    # 创建图并去除自环
    G = nx.from_numpy_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # 节点的数量
    num_nodes = len(features_matrix)
    
    # 节点颜色（特征矩阵的第一列）
    node_colors = features_matrix[:, 0]
    
    # 节点大小（特征矩阵的第二列，放大系数为100）
    node_sizes = np.clip(features_matrix[:, 1] * 100, 10, 1000)  # 防止节点大小过小或过大
    
    # 计算边的宽度和颜色
    edge_weights = adj_matrix[adj_matrix > 0]
    edge_widths = np.clip(edge_weights * 5, 0.1, 10)  # 边宽范围 [0.1, 10]
    
    # 使用规范化映射边的颜色
    norm = mpl.colors.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
    edge_colors = plt.cm.RdYlGn(norm(edge_weights.flatten()))  # 使用色图规范化边的权重
    
    # 设置图的布局
    pos = nx.spring_layout(G)  # 使用Spring布局计算节点位置
    
    # 检查 pos 中是否有无效坐标
    for node, coords in pos.items():
        if any(map(lambda x: x != x, coords)):  # 检查是否有 NaN
            print(f"Invalid coordinates for node {node}: {coords}")
    
    # 绘制图形
    plt.figure(figsize=(8, 8))
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, cmap=plt.cm.viridis)

    # 绘制边
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
    
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos)
    
    plt.title("Graph Visualization with Edge Weights")
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])  # 无需传入实际数据，空数组即可
    plt.colorbar(sm, label="Edge Weight")
    
    # 保存图像
    plt.savefig(path, dpi=300, bbox_inches='tight')


def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter value:\n{param.data}")  # 打印参数的值
        print(f"Gradient:\n{param.grad}")  # 打印参数的梯度
        print("-" * 50)  # 分隔线，用来分隔每个参数
def main(args):
    loss_func = Loss(args)
    inputs, mask = get_target(args)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args, inputs, mask)

    optimizer = PainterOptimizer(args, renderer)
    
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss = 100, 100
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-5
    terminate = False

    renderer.set_random_noise(0)
    img = renderer.init_image(stage=0)
    optimizer.init_optimizers()
    
    # not using tdqm for jupyter demo
    if args.display:
        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))
    gcn = GCN(8,16,8).to(args.device)
    gcn_opt = torch.optim.Adam(gcn.parameters(), lr=0.01)

    for epoch in epoch_range:
        # print(epoch)
        if not args.display:
            epoch_range.refresh()
        renderer.set_random_noise(epoch)
        if args.lr_scheduler:
            optimizer.update_lr(counter)

        start = time.time()
        gcn.train()
        optimizer.zero_grad_()
        gcn_opt.zero_grad()
        # print(renderer.control_points_set)
        out = gcn(renderer.get_feature_matrix(),renderer.get_adj_matrix())
        out = out.view(16,4,2)
        print(out)
        for points in out:
            for point in points:
                if point[0] < 0 or point[0] < 0 or point[0] > 1 or point[1] > 1:
                    print(point)
        sketches = render_img(out,renderer).to(args.device)
        sketch = renderer.get_image().to(args.device)
        
        # print(sketches)
        # print("___________________")
        # print(sketch)
        losses_dict = loss_func(sketches, inputs.detach(
        ), renderer.get_color_parameters(), renderer, counter, optimizer)
        loss = sum(list(losses_dict.values()))
        loss.backward()
        gcn_opt.step()
        optimizer.step_()
        if epoch % args.save_interval == 0:
            # print_model_parameters(gcn)
            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            # renderer.save_svg(f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
            renderer.save_svgs(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
            render_matrix(renderer, f"{args.output_dir}/adj_logs/iter{epoch}.png")
            utils.plot_batch(inputs,render_colored_img(renderer), f"{args.output_dir}/color_logs", counter,use_wandb=args.use_wandb, title=f"iter{epoch}.jpg",single=True)
        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
                ), renderer.get_points_parans(), counter, optimizer, mode="eval")
                loss_eval = sum(list(losses_dict_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                for k in losses_dict_eval.keys():
                    if k not in configs_to_save.keys():
                        configs_to_save[k] = []
                    configs_to_save[k].append(losses_dict_eval[k].item())
                if args.clip_fc_loss_weight:
                    if losses_dict_eval["fc"].item() < best_fc_loss:
                        best_fc_loss = losses_dict_eval["fc"].item(
                        ) / args.clip_fc_loss_weight
                        best_iter_fc = epoch
                # print(
                #     f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")

                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        terminate = False
                        utils.plot_batch(
                            inputs, sketches, args.output_dir, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                        renderer.save_svg(args.output_dir, "best_iter")

                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_loss_fc"] = best_fc_loss
                    wandb_dict = {"delta": cur_delta,
                                  "loss_eval": loss_eval.item()}
                    for k in losses_dict_eval.keys():
                        wandb_dict[k + "_eval"] = losses_dict_eval[k].item()
                    wandb.log(wandb_dict, step=counter)

                if abs(cur_delta) <= min_delta:
                    if terminate:
                        break
                    terminate = True

        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                                 args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            for k in losses_dict.keys():
                wandb_dict[k] = losses_dict[k].item()
            wandb.log(wandb_dict, step=counter)

        counter += 1

    renderer.save_svg(args.output_dir, "final_svg")
    path_svg = os.path.join(args.output_dir, "best_iter.svg")
    utils.log_sketch_summary_final(
        path_svg, args.use_wandb, args.device, best_iter, best_loss, "best total")

    return configs_to_save

if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()
