"""
Provides options via the command line to perform project tasks.
* `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
* `--transform`: transform dataset to the HDF5 file
* `--cv2`: visualize sample from transformed dataset
* `--image`: predict a single image with the source parameter
* `--train`: train model with the source argument
* `--test`: evaluate and predict model with the source argument
* `--norm_accentuation`: discard accentuation marks in the evaluation
* `--norm_punctuation`: discard punctuation marks in the evaluation
* `--epochs`: number of epochs
* `--batch_size`: number of batches
"""
from pathlib import Path

import torch

import numpy as np
import argparse
import cv2
import h5py
import os
import string
import torchvision.transforms as T

from data import preproc as pp, evaluation
from data.generator import DataGenerator, Tokenizer
from data.reader import Dataset
from network.model import make_model
from engine import single_image_inference, LabelSmoothing, run_epochs,get_memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)

    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--image", type=str, default="")

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

    parser.add_argument("--norm_accentuation", action="store_true", default=False)
    parser.add_argument("--norm_punctuation", action="store_true", default=False)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    
    args = parser.parse_args()

    raw_path = os.path.join("..", "raw", args.source)
    source_path = os.path.join("..", "data", f"{args.source}.hdf5")
    output_path = os.path.join("..", "output", args.source,)
    target_path = os.path.join(output_path, "checkpoint_weights.pt")

    input_size = (1024, 128, 1)
    max_text_length = 128
    charset_base = string.printable[:95]
    tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)            

    if args.transform:
        print(f"{args.source} dataset will be transformed...")

        ds = Dataset(source=raw_path, name=args.source)
        ds.read_partitions()

        print("Partitions will be preprocessed...")
        ds.preprocess_partitions(input_size=input_size)

        print("Partitions will be saved...")
        os.makedirs(os.path.dirname(source_path), exist_ok=True)

        for i in ds.partitions:
            with h5py.File(source_path, "a") as hf:
                hf.create_dataset(f"{i}/dt", data=ds.dataset[i]['dt'], compression="gzip", compression_opts=9)
                hf.create_dataset(f"{i}/gt", data=ds.dataset[i]['gt'], compression="gzip", compression_opts=9)
                print(f"[OK] {i} partition.")

        print(f"Transformation finished.")

    elif args.image:
        
        img = pp.preprocess(args.image, input_size=input_size)
        
        #making image compitable with resnet
        img = np.repeat(img[..., np.newaxis],3, -1)
        x_test = pp.normalization(img)

        model = make_model(tokenizer.vocab_size, hidden_dim=256, nheads=4,
                 num_encoder_layers=4, num_decoder_layers=4)
        device = torch.device(args.device)
        model.to(device)
        transform = T.Compose([
                T.ToTensor()])
                

        if os.path.exists(target_path):
            model.load_state_dict(torch.load(target_path))            
        else:            
            print('No model checkpoint found')
        
        prediction = single_image_inference(model, x_test, tokenizer, transform, device)
        
        print("\n####################################")
        print("predicted text is: {}".format(prediction))
        cv2.imshow("Image ", cv2.imread(args.image))
        print("\n####################################")
        cv2.waitKey(0)

    else:
        assert os.path.isfile(source_path) or os.path.isfile(target_path)
        os.makedirs(output_path, exist_ok=True)
        
        if args.train:
            
            device = torch.device(device)
            model = make_model(tokenizer.vocab_size, hidden_dim=256, nheads=4,
                     num_encoder_layers=4, num_decoder_layers=4)
            model.to(device)
            
            train_loader = torch.utils.data.DataLoader(DataGenerator(source_path,charset_base,max_text_length,'train',transform), batch_size=args.batch_size, shuffle=False, num_workers=2)
            val_loader = torch.utils.data.DataLoader(DataGenerator(source_path,charset_base,max_text_length,'valid',transform), batch_size=args.batch_size, shuffle=False, num_workers=2)

            criterion = LabelSmoothing(size=tokenizer.vocab_size, padding_idx=0, smoothing=0.1)
            criterion.to(device)
            lr = args.lr # learning rate
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=.0004)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

            run_epochs(model, criterion, optimizer, scheduler, train_loader, val_loader, args.epochs, tokenizer, target_path)                


        elif args.test:
            
            model.eval()
            predicts = []
            gt = []
            tes_loader = torch.utils.data.DataLoader(DataGenerator(source_path,charset_base,max_text_length,'train',transform), batch_size=1, shuffle=False, num_workers=2)
            with torch.no_grad():
                for batch in tes_loader:
                    src, trg = batch
                    src, trg = src.cuda(), trg.cuda()
                    
                    memory = get_memory(model,src.float())
                    out_indexes = [tokenizer.chars.index('SOS'), ]
                    for i in range(max_text_length):
                        mask = model.generate_square_subsequent_mask(i+1).to('cuda')
                        trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
                        output = model.vocab(model.transformer.decoder(model.query_pos(model.decoder(trg_tensor)), memory,tgt_mask=mask))
                        out_token = output.argmax(2)[-1].item()
                        out_indexes.append(out_token)
                        # print(output.shape)
                        if out_token == 3:
                            break
                        
                    predicts.append(tokenizer.decode(out_indexes))
                    gt.append(tokenizer.decode(trg.flatten(0,1)))

            predicts = list(map(lambda x : x.replace('SOS','').replace('EOS',''),predicts))
            gt = list(map(lambda x : x.replace('SOS','').replace('EOS',''),gt))
            
            evaluate = evaluation.ocr_metrics(predicts=predicts,
                                              ground_truth=gt,
                                              norm_accentuation=args.norm_accentuation,norm_punctuation=args.norm_punctuation)
            print("Calculate Character Error Rate {}, Word Error Rate {} and Sequence Error Rate {}".format(evaluate[0],evaluate[1],evaluate[2]))
