
import argparse
import open3d as o3d
import os
import json
import sys

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(pwd, '..'))


class ReaderWithCallback:

    def __init__(self, input, output):
        self.flag_exit = False
        self.flag_play = True
        self.input = input
        self.output = output
        self.reader = o3d.io.AzureKinectMKVReader()
        self.reader.open(self.input)
        if not self.reader.is_opened():
            raise RuntimeError("Unable to open file {}".format(args.input))

    def run(self):

        abspath = os.path.abspath(self.output)
        metadata = self.reader.get_metadata()
        o3d.io.write_azure_kinect_mkv_metadata(
            '{}\intrinsic.json'.format(abspath), metadata)
        print('{}\intrinsic.json'.format(abspath))
        import pdb; pdb.set_trace()
        self.reader.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect mkv reader.')
    parser.add_argument('--input',
                        type=str,
                        required=True,
                        help='input mkv file')
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        exit()

    output_dir = os.path.basename(args.input).replace('.mkv', '')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    reader = ReaderWithCallback(args.input, output_dir)
    reader.run()
