import torch
from collections import defaultdict
import numpy as np
import math
from PIL import Image
from torch.nn.functional import log_softmax, softmax
from src.models.components.translate.beam import Beam
import torchvision.transforms as transforms
class Predictor():
    def __init__(self, model, vocab) -> None:
        self.model = model
        self.vocab = vocab

    #________________________PROCESS_INPUT______________________
    def resize(self,w, h, expected_height, image_min_width, image_max_width):
        new_w = int(expected_height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w/round_to)*round_to
        new_w = max(new_w, image_min_width)
        new_w = min(new_w, image_max_width)

        return new_w, expected_height

    def process_image(self,image, image_height=32, image_min_width=32, image_max_width=512):
        img = image.convert('RGB')

        w, h = img.size
        new_w, image_height = self.resize(w, h, image_height, image_min_width, image_max_width)

        img = img.resize((new_w, image_height), Image.LANCZOS)

        img = np.asarray(img)#.transpose(2, 0, 1)
        # img = img/255 # not necessary because transforms.ToTensor() automatically do it
        return img
    
    def process_input(self,image, image_height=32, image_min_width=32, image_max_width=512):
        img = self.process_image(image, image_height, image_min_width, image_max_width)
        
        # img = torch.FloatTensor(img)
        img  = transforms.ToTensor()(img)
        img = img[np.newaxis, ...]
        return img

    #_________________________TRANSLATE____________________________

    def translate(self, img, max_seq_length = 128, sos_token = 1, eos_token = 2):
        self.model.eval()

        device = img.device

        with torch.no_grad():
            src = self.model.cnn(img)
            memory = self.model.transformer.forward_encoder(src)

            translated_sentence = [[sos_token]*len(img)]
            char_probs = [[1]*len(img)]

            max_length = 0

            while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

                tgt_inp = torch.LongTensor(translated_sentence).to(device)

#                output = model(img, tgt_inp, tgt_key_padding_mask=None)
#                output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
                output, memory = self.model.transformer.forward_decoder(tgt_inp, memory)
                output = softmax(output, dim=-1)
                output = output.to('cpu')

                values, indices  = torch.topk(output, 5)

                indices = indices[:, -1, 0]
                indices = indices.tolist()

                values = values[:, -1, 0]
                values = values.tolist()
                char_probs.append(values)

                translated_sentence.append(indices)   
                max_length += 1

                del output

            translated_sentence = np.asarray(translated_sentence).T

            char_probs = np.asarray(char_probs).T
            char_probs = np.multiply(char_probs, translated_sentence>3)
            char_probs = np.sum(char_probs, axis=-1)/(char_probs>0).sum(-1)

        return translated_sentence, char_probs
    
    def batch_translate_beam_search(self,img, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: NxCxHxW
        self.model.eval()
        device = img.device
        sents = []

        with torch.no_grad():
            src = self.model.cnn(img)
            print(src.shap)
            memories = self.model.transformer.forward_encoder(src)
            for i in range(src.size(0)):
#                memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
                memory = self.model.transformer.get_memory(memories, i)
                sent = self.beamsearch(memory, device, beam_size, candidates, max_seq_length, sos_token, eos_token)
                sents.append(sent)

        sents = np.asarray(sents)

        return sents
   
    def translate_beam_search(self, img, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
        # img: 1xCxHxW
        self.model.eval()
        device = img.device

        with torch.no_grad():
            src = self.model.cnn(img)
            memory = self.model.transformer.forward_encoder(src) #TxNxE
            sent = self.beamsearch(memory,device, beam_size, candidates, max_seq_length, sos_token, eos_token)

        return sent

    def beamsearch(self,memory, device, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):    
        # memory: Tx1xE
        self.model.eval()

        beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token, end_token_id=eos_token)

        with torch.no_grad():
    #        memory = memory.repeat(1, beam_size, 1) # TxNxE
            memory = self.model.transformer.expand_memory(memory, beam_size)

            for _ in range(max_seq_length):

                tgt_inp = beam.get_current_state().transpose(0,1).to(device)  # TxN
                decoder_outputs, memory = self.model.transformer.forward_decoder(tgt_inp, memory)

                log_prob = log_softmax(decoder_outputs[:,-1, :].squeeze(0), dim=-1)
                beam.advance(log_prob.cpu())

                if beam.done():
                    break

            scores, ks = beam.sort_finished(minimum=1)

            hypothesises = []
            for i, (times, k) in enumerate(ks[:candidates]):
                hypothesis = beam.get_hypothesis(times, k)
                hypothesises.append(hypothesis)

        return [1] + [int(i) for i in hypothesises[0][:-1]]
    #______________________________________PREDICTION___________
    def predict(self, img,  device, return_prob=False):
        img = self.process_input(img)  
        print(img.shape)      
        img = img.to(device=device)


        # if self.config['predictor']['beamsearch']:
        #     sent = self.translate_beam_search(img)
        #     s = sent
        #     prob = None
        # else:
        #     s, prob = self.translate(img)
        #     s = s[0].tolist()
        #     prob = prob[0]
        s, prob = self.translate(img)
        s = s[0].tolist()
        prob = prob[0]
        s = self.vocab.decode(s)
        
        if return_prob:
            return s, prob
        else:
            return s

    def predict_batch(self, imgs, device, return_prob=False):
        bucket = defaultdict(list)
        bucket_idx = defaultdict(list)
        bucket_pred = {}
        
        sents, probs = [0]*len(imgs), [0]*len(imgs)

        for i, img in enumerate(imgs):
            img = self.process_input(img)      
        
            bucket[img.shape[-1]].append(img)
            bucket_idx[img.shape[-1]].append(i)


        for k, batch in bucket.items():
            batch = torch.cat(batch, 0).to(device=device)
            s, prob = self.translate(batch)
            prob = prob.tolist()

            s = s.tolist()
            s = self.vocab.batch_decode(s)

            bucket_pred[k] = (s, prob)


        for k in bucket_pred:
            idx = bucket_idx[k]
            sent, prob = bucket_pred[k]
            for i, j in enumerate(idx):
                sents[j] = sent[i]
                probs[j] = prob[i]
   
        if return_prob: 
            return sents, probs
        else: 
            return sents
    