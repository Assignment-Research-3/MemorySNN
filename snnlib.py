from glob import glob
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cupy as cp

IMAGE_SIZE = 32  # size of images (width & height)
IMAGE_PIXEL_THRESHOLD = .05  # threshold for pixel values

def gram_schmidt_process(V: np.ndarray) -> np.ndarray:
    n, N = V.shape
    U = np.zeros((n, N), dtype=np.float64)
    U[0] = V[0] / np.linalg.norm(V[0])
    for i in range(1, n):    
        a = np.zeros((N,), dtype=np.float64)
        for j in range(i):
            a += np.dot(V[i], U[j]) * U[j]
        U[i] = V[i] - a
        U[i] = U[i] / np.linalg.norm(U[i])
    return U

def encode_images(img_paths: str, tags: np.ndarray, means, stds) -> np.ndarray:
    """ encode images to neural states (memory components) """

    d = IMAGE_SIZE
    num_comp = len(img_paths)
    dim = (d ** 2) * tags.shape[1]
    mem_comp = np.zeros((num_comp, dim))
    for i, (img_path, tag) in enumerate(zip(img_paths, tags)):
        img = Image.open(img_path).convert("L")
        img = img.resize((d, d))
        img_values = np.array(img) / 255
        img_values = np.clip(means[i] + (img_values - img_values.mean()) / img_values.std() * stds[i], 0, 1)
        img_values = (img_values - 0.5) * (2 * IMAGE_PIXEL_THRESHOLD)
        img_values = np.reshape(img_values, (d ** 2,))
        mem_comp[i] = np.outer(tag, img_values).reshape(-1)  # tensor product binding
    return mem_comp

def encode_images_no_tag(img_paths: str, means, stds) -> np.ndarray:
    """ encode images to neural states (memory components) """

    d = IMAGE_SIZE
    num_comp = len(img_paths)
    dim = (d ** 2) * 6
    mem_comp = np.zeros((num_comp, dim))
    for i, (img_path, tag) in enumerate(zip(img_paths, tags)):
        img = Image.open(img_path).convert("L")
        img = img.resize((d, d))
        img_values = np.array(img) / 255
        img_values = np.clip(means[i] + (img_values - img_values.mean()) / img_values.std() * stds[i], 0, 1)
        img_values = (img_values - 0.5) * (2 * IMAGE_PIXEL_THRESHOLD)
        img_values = np.reshape(img_values, (d ** 2,))
        mem_comp[i] = img_values.reshape(-1)  # tensor product binding
    return mem_comp

def encode_box_images(img_paths: str, tags: np.ndarray, mean_, std_) -> np.ndarray:
    """ encode images to neural states (memory components) """

    d = IMAGE_SIZE
    num_comp = len(img_paths)
    dim = (d ** 2) * tags.shape[1]
    mem_comp = np.zeros((num_comp, dim))
    img = Image.open(img_paths[0]).convert("L")
    img = img.resize((d, d))
    img_values = np.array(img) / 255
    img_values = np.clip(mean_ + (img_values - img_values.mean()) / img_values.std() * std_, 0, 1)
    img_values[10:22, 10:22] = img_values[10:22, 10:22].mean()
    img_values = (img_values - 0.5) * (2 * IMAGE_PIXEL_THRESHOLD)
    img_values = np.reshape(img_values, (d ** 2,))
    for i in range(5):
        mem_comp[i] = np.outer(tags[i], img_values).reshape(-1)  # tensor product binding
    return mem_comp

def encode_box_images_no_tag(img_paths: str, mean_, std_) -> np.ndarray:
    """ encode images to neural states (memory components) """

    d = IMAGE_SIZE
    num_comp = len(img_paths)
    dim = (d ** 2) * 6
    mem_comp = np.zeros((num_comp, dim))
    img = Image.open(img_paths[0]).convert("L")
    img = img.resize((d, d))
    img_values = np.array(img) / 255
    img_values = np.clip(mean_ + (img_values - img_values.mean()) / img_values.std() * std_, 0, 1)
    img_values[10:22, 10:22] = img_values[10:22, 10:22].mean()
    img_values = (img_values - 0.5) * (2 * IMAGE_PIXEL_THRESHOLD)
    img_values = np.reshape(img_values, (d ** 2,))
    for i in range(5):
        mem_comp[i] = img_values.reshape(-1)  # tensor product binding
    return mem_comp

def decode_neural_state(spike_count: np.ndarray, tags: np.ndarray) -> plt.Figure:
    print(spike_count.max(), spike_count.min())
    spike_count = (spike_count - spike_count.mean()) / spike_count.std()
    d = IMAGE_SIZE
    n = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(1 * n, 1))
    if len(tags) == 1:
        axes = [axes]
    for tag, ax in zip(tags, axes):
        ax: plt.Axes
        decoded = tag @ np.reshape(spike_count, (-1, d ** 2))
        decoded = np.reshape(decoded, (d, d))
        ax.imshow(decoded, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig

from dataclasses import dataclass

@dataclass
class NeuronProp:
    V_rest: float
    V_th: float
    tau_m: float
    E_ex: float
    tau_e: float

    def to_cupy(self, dt):
        return CUDANeuronProp(
            V_rest=cp.asarray(self.V_rest),
            V_th=cp.asarray(self.V_th),
            tau_m=cp.asarray(self.tau_m / dt),
            E_ex=cp.asarray(self.E_ex),
            tau_e=cp.asarray(self.tau_e / dt),
        )

@dataclass
class SynapseProp:
    tau_pre: float
    tau_post: float
    dA_pre: float
    dA_post: float
    g_max: float
    w_ratio: float

    def to_cupy(self, dt):
        return CUDASynapseProp(
            tau_pre=cp.asarray(self.tau_pre / dt),
            tau_post=cp.asarray(self.tau_post / dt),
            dA_pre=cp.asarray(self.dA_pre),
            dA_post=cp.asarray(self.dA_post),
            g_max=cp.asarray(self.g_max),
            w_ratio=cp.asarray(self.w_ratio)
        )

@dataclass
class CUDANeuronProp:
    V_rest: cp.ndarray
    V_th: cp.ndarray
    tau_m: cp.ndarray
    E_ex: cp.ndarray
    tau_e: cp.ndarray

@dataclass
class CUDASynapseProp:
    tau_pre: cp.ndarray
    tau_post: cp.ndarray
    dA_pre: cp.ndarray
    dA_post: cp.ndarray
    g_max: cp.ndarray
    w_ratio: cp.ndarray

class InputGenerator:
    def __init__(self, N):
        self.N = N

    def step(self, dt: float) -> np.ndarray:
        return np.zeros((self.N, ))
    
class SineWaveInputGenerator(InputGenerator):
    def __init__(self, arr: np.ndarray, omega: float):
        self.xi = cp.asarray(np.linspace(0, np.pi, arr.shape[0] + 1)[:-1])
        self.t = 0.
        self.omega = cp.asarray(omega)
        self.arr = cp.asarray(arr)
    
    def step(self, dt: float) -> cp.ndarray:
        self.t += dt
        return cp.sin(cp.asarray(self.t) * self.omega - self.xi) @ self.arr

class MemorySNN:
    def __init__(self, N: int, neuron_prop: NeuronProp, synapse_prop: SynapseProp):
        self.N = N
        self.nprop = neuron_prop
        self.sprop = synapse_prop
        self.W = np.zeros((N, N))
        self.V = np.full((N, ), self.nprop.V_rest)
        self.ge = np.zeros((N, ))
        self.Apre = np.zeros((N, ))
        self.Apost = np.zeros((N, ))

    def clear(self):
        self.V = np.full((self.N, ), self.nprop.V_rest)
        self.ge = np.zeros((self.N, ))
        self.Apre = np.zeros((self.N, ))
        self.Apost = np.zeros((self.N, ))

    def save_W(self, path: str) -> None:
        np.save(path, self.W)

    def load_W(self, path: str) -> None:
        self.W = np.load(path)

    def learn_memory(self, input: InputGenerator, steps: int, dt: float) -> np.ndarray:
        """ learning memory components """
        V = cp.asarray(self.V)
        ge = cp.asarray(self.ge)
        Apre = cp.asarray(self.Apre)
        Apost = cp.asarray(self.Apost)
        nprop = self.nprop.to_cupy(dt)
        sprop = self.sprop.to_cupy(dt)
        W = cp.asarray(self.W)
        spike_count = np.zeros((steps,), dtype=int)

        for t in tqdm(range(steps)):
            V += (ge * (nprop.E_ex - V) + input.step(dt)) / nprop.tau_m
            ge += -ge / nprop.tau_e 
            Apre += -Apre / sprop.tau_pre
            Apost += -Apost / sprop.tau_post
            spiked = V > nprop.V_th
            V[spiked] = nprop.V_rest
            ge += W[spiked, :].sum(axis=0) * sprop.w_ratio
            Apre[spiked] += sprop.dA_pre
            W[spiked, :] = cp.clip(W[spiked, :] + Apost, 0, sprop.g_max)
            Apost[spiked] += sprop.dA_post
            W[:, spiked] = cp.clip(W[:, spiked] + Apre.reshape(-1, 1), 0, sprop.g_max)
            spike_count[t] = spiked.sum().get()

        self.V = V.get()
        self.ge = ge.get()
        self.Apre = Apre.get()
        self.Apost = Apost.get()
        self.W = W.get()
        plt.scatter(np.arange(len(spike_count)), spike_count)
        plt.savefig('result/learn_spike_count.png')
        plt.close()

    def simulate(self, input: InputGenerator, steps: int, dt: float) -> np.ndarray:
        """ simulating SNN without initialization """
        spike_count = cp.asarray(np.zeros((self.N, )))
        V = cp.asarray(self.V)
        ge = cp.asarray(self.ge)
        Apre = cp.asarray(self.Apre)
        Apost = cp.asarray(self.Apost)
        nprop = self.nprop.to_cupy(dt)
        sprop = self.sprop.to_cupy(dt)
        W = cp.asarray(self.W)

        for t in tqdm(range(steps)):
            V += (ge * (nprop.E_ex - V) + input.step(dt)) / nprop.tau_m
            ge += -ge / nprop.tau_e 
            Apre += -Apre / sprop.tau_pre
            Apost += -Apost / sprop.tau_post
            spiked = V > nprop.V_th
            V[spiked] = nprop.V_rest
            ge += W[spiked, :].sum(axis=0) * sprop.w_ratio
            Apre[spiked] += sprop.dA_pre
            Apost[spiked] += sprop.dA_post
            spike_count += spiked

        self.V = V.get()
        self.ge = ge.get()
        self.Apre = Apre.get()
        self.Apost = Apost.get()

        return spike_count.get()
    
    def retrieve_from_cue(self, cue: np.ndarray, tags: np.ndarray, omega) -> None:
        """ retrieval from cue signal """
        self.clear()
        igen = SineWaveInputGenerator(cue, omega)

        for i in range(10):
            spike_count = self.simulate(igen, 100, 0.01)
            decode_neural_state(spike_count, tags).savefig(f'result/retrieved_cue_{i}')
            plt.close()

import zipfile
import os
import smtplib
from email.message import EmailMessage

def zip_directory(folder_path, zip_name="output.zip"):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                # 압축 내부에서의 경로를 상대경로로 설정
                arcname = os.path.relpath(full_path, start=folder_path)
                zipf.write(full_path, arcname=arcname)
    return zip_name

def send_email_with_zip(zip_path, subject, body, sender, receiver, smtp_server, smtp_port, password):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver
    msg.set_content(body)

    # ZIP 파일 첨부
    with open(zip_path, "rb") as f:
        data = f.read()
        msg.add_attachment(data, maintype="application", subtype="zip", filename=os.path.basename(zip_path))

    # 메일 전송
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)
    print("✅ 메일 전송 완료")
