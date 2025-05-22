import torch
import torchaudio
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import IPython
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

#匯入測試音檔
SPEECH_FILE = "d:/大四下專題/audio2.wav"

#建立Wav2Vec2 模型，使用預訓練好的針對ASR加強的權重
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

#使用預訓練好的權重創建模型
model = bundle.get_model().to(device)

#得到測試檔案的波型，取樣綠
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

#將音檔轉換為模型的取樣率
if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

#提取音檔的特徵
with torch.inference_mode():
    features, _ = model.extract_features(waveform)

#圖形化特徵
fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu(), interpolation="nearest")
    ax[i].set_title(f"Feature from transformer layer {i+1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis)")
fig.tight_layout()
fig.show()

#分類特徵
with torch.inference_mode():
    emission, _ = model(waveform)

#圖形化特徵分類
plt.imshow(emission[0].cpu().T, interpolation="nearest")
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.tight_layout()
plt.show()

#解碼產生文字檔
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
    
decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
print(transcript)
IPython.display.Audio(SPEECH_FILE)
