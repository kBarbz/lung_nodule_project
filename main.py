import pre_process
import dataset_annotation
import efficientdet_repo
import argparse
from glob import glob
from config.config import DATASET_PATH


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficient_det')
    ap.add_argument('-p', '--project', type=str, default='luna16', help='project file that contains parameters')
    ap.add_argument('--weights', type=str, default=None, help='/path/to/weights')
    ap.add_argument('--nms_threshold', type=float, default=0.5,
                    help='nms threshold, don\'t change it if not for testing purposes')
    ap.add_argument('--cuda', type=boolean_string, default=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--float16', type=boolean_string, default=False)
    ap.add_argument('--override', type=boolean_string, default=True,
                    help='override previous bbox results file if exists')
    ap.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    ap.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    ap.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy datasets')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    ap.add_argument('--num_epochs', type=int, default=500)
    ap.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    ap.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    ap.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    ap.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    ap.add_argument('--data_path', type=str, default=f'{DATASET_PATH}/prepared_data/',
                        help='the root folder of datasets')
    ap.add_argument('--log_path', type=str, default='logs/')
    ap.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    ap.add_argument('--saved_path', type=str, default='logs/')
    ap.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')
    ap.add_argument('--skip_pre', type=boolean_string, default=False,
                    help='Skip preprocessing')
    ap.add_argument('--skip_train', type=boolean_string, default=False,
                    help='Skip preprocessing')

    args = ap.parse_args()
    if not args.skip_pre:
        pre_process.process.run()
        dataset_annotation.main.run()
    if not args.skip_train:
        efficientdet_repo.train.run()
    weights_path = [args.weights] if args.weights else glob(rf'logs/luna16/efficientdet-d{args.compound_coef}*.pth')
    efficientdet_repo.coco_eval.run(sorted(weights_path, key=lambda x: int(x.split('_')[1]))[-1])