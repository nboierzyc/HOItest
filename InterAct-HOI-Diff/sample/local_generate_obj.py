# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip, load_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.behave.scripts.motion_process import recover_from_ric
import data_loaders.behave.utils.paramUtil as paramUtil
from data_loaders.behave.utils.plot_script import plot_3d_motion
from utils.visualize_marker import plot_markers
import shutil
from data_loaders.tensors import collate, afford_collate
from trimesh import Trimesh
import trimesh
from scipy.spatial.transform import Rotation
from data_loaders.behave.data.dataset import text_to_object
from visualize.vis_utils import simplified_mesh
from model.hoi_diff import HOIDiff as used_model
from model.afford_est import AffordEstimation
from diffusion.gaussian_diffusion import LocalMotionDiffusion, AffordDiffusion
from sample.condition import Guide_Contact


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 300 
    fps = 30
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)

    sigmoid = torch.nn.Sigmoid()
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples


    print('Loading Motion dataset...')
    data = load_motion_dataset(args, max_frames, n_frames)

    total_num_samples = args.num_samples * args.num_repetitions




    print("Creating motion model and diffusion...")
    motion_model, motion_diffusion = load_model(args, data, dist_util.dev(), ModelClass=used_model, DiffusionClass=LocalMotionDiffusion, diff_steps=1000,model_path=args.model_path)

        
    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        n_frames = int(args.motion_length * fps)
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            obj_points, obj_normals, obj_name, obj_bps = text_to_object(texts)

            collate_args = [dict(arg, text=txt, obj_points=points, obj_normals=normals, seq_name = name,obj_bps=bps) for arg, txt, points, normals, name,bps in zip(collate_args, texts, obj_points, obj_normals, obj_name, obj_bps)]
   

        _, model_kwargs = collate(collate_args)

    # input to cuda
    model_kwargs['y']['obj_points'] = model_kwargs['y']['obj_points'].to(dist_util.dev())
    model_kwargs['y']['obj_normals'] = model_kwargs['y']['obj_normals'].to(dist_util.dev())
    model_kwargs['y']['obj_bps'] = model_kwargs['y']['obj_bps'].to(dist_util.dev())



    all_motions = []
    all_motions_obj = []
   
    all_lengths = []
    all_text = []
    all_obj_name = []
    all_obj_points = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param


        if not args.skip_first_stage:
            afford_sample_fn = afford_diffusion.p_sample_loop

            afford_sample = afford_sample_fn(
                afford_model,
                (args.batch_size, 4, afford_model.nfeats, 8),    # + 6 object pose
                # clip_denoised=False,
                clip_denoised=not args.predict_xstart,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
                cond_fn=None,
            )

            afford_sample[:,3:] = sigmoid(afford_sample[:,3:])





        if args.guidance:
            guide_fn_contact = Guide_Contact(
                                            inv_transform_th=data.dataset.t2m_dataset.inv_transform_th,
                                            mean=data.dataset.t2m_dataset.mean,
                                            std=data.dataset.t2m_dataset.std,
                                            classifiler_scale=args.classifier_scale,
                                            use_global=data.dataset.use_global,
                                            batch_size=afford_sample.shape[0],
                                            afford_sample = afford_sample
                                            )
        else: 
            guide_fn_contact = None


        sample_fn = motion_diffusion.p_sample_loop

        sample = sample_fn(
            motion_model,
            (args.batch_size, motion_model.njoints + 6, motion_model.nfeats, n_frames),    # + 6 object pose
            clip_denoised=False,
            # clip_denoised=not args.predict_xstart,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn=guide_fn_contact,
        )

        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()

        sample_obj = sample[..., 553:]
        sample_obj = sample_obj.permute(0, 1, 3, 2)
        sample = sample[..., :553]
        n_markers = 77

        
        sample = sample[:,:,:,:n_markers*3]
        sample = sample.reshape(sample.shape[0], sample.shape[1], sample.shape[2], n_markers, 3)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)


        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]


        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        all_motions_obj.append(sample_obj.cpu().numpy())
        all_obj_points.append(model_kwargs['y']['obj_points'].cpu().numpy())
        all_obj_name += model_kwargs['y']['seq_name']


        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_motions_obj = np.concatenate(all_motions_obj, axis=0)[:total_num_samples]
    all_obj_points = np.concatenate(all_obj_points, axis=0)
    all_obj_name = all_obj_name[:total_num_samples]


    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')


    save_dict = {'motion': all_motions, 'motion_obj':all_motions_obj, 'text': all_text, 'lengths': all_lengths,
                'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions, 'obj_name': all_obj_name}

    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, save_dict)

    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            motion_obj = all_motions_obj[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length,0,:]

            vertices = all_obj_points[rep_i*args.batch_size + sample_i]

            if is_using_data:
                obj_name = all_obj_name[rep_i*args.batch_size + sample_i].split('_')[2]
            else:
                obj_name = all_obj_name[rep_i*args.batch_size + sample_i]


            mesh_path = os.path.join("./data/behave/objects", f'{obj_name}/sample_points.npy')
            vertices = np.load(mesh_path)


            # transform
            angle, trans = motion_obj[:, :3].transpose(1,0), motion_obj[:, 3:].transpose(1,0)
            rot = Rotation.from_rotvec(angle.transpose(1, 0)).as_matrix()
            obj_points = np.matmul(vertices[np.newaxis], rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + trans.transpose(1, 0)[:, np.newaxis]


            
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)



            # plot_3d_motion(animation_save_path, skeleton, motion, obj_points, hc_mask=None, oc_mask=None, title=caption, fps=fps)
            plot_markers(animation_save_path, motion, obj_points)

            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(args, out_path,
                                               row_print_template, all_print_template, row_file_template, all_file_template,
                                               caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template



def load_afford_dataset(args, training_stage=1):
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=1,
        split='test',
        hml_mode='text_only',
        training_stage=training_stage)
    data = get_dataset_loader(data_conf)
    return data

def load_motion_dataset(args, max_frames, n_frames, training_stage=3): 
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split='test',
        hml_mode='text_only',
        training_stage=training_stage)
    data = get_dataset_loader(data_conf)
    data.fixed_length = n_frames
    return data


    

if __name__ == "__main__":
    main()
