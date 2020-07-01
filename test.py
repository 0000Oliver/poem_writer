import torch
def generate(model,start_words,ix2word,word2ix):
    result = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<STATT>']]).view(1,1).long()
    hidden = None
    model.eval()
    with torch.no_grad():
        for i in range(Config.max_gen_len):
            output,hidden = model(input,hidden)
            # 如果在给定的句首中，input为句首中的下一个字
            if i<start_words_len:
                w = result[i]
                input= input.data.new([word2ix[w]]).view(1,1)
            # 否则将output作为下一个input进行
            else:
                top_index = output.data[0].topk(1)(1)[0].item()
                w = ix2word[top_index]
                result.append(w)
                input = input.data.new([top_index]).view(1,1)
            if w =='<EOP>':
                del result[-1]
                break
        return result
