from snnlib import *
import os, sys

try:
  os.mkdir('result')
except:
  pass

cp.cuda.Device(1).use()

img_paths = glob("images/*.png")
store_img_paths = [p for p in img_paths if "forest" not in p]  # forest 사진 제외 악기 사진 5개 사용
store_img_paths.sort()

n = len(img_paths)
print(f"Number of images: {n}")

nprop_default = NeuronProp(V_rest=0.,
                           V_th=0.06,
                           tau_m=0.02,
                           E_ex=0.06,
                           tau_e=0.02)

sprop_default = SynapseProp(tau_pre=0.02,
                            tau_post=0.02,
                            dA_pre=0.01,
                            dA_post= -0.0105,
                            g_max=10.,
                            w_ratio=1/310)

run_count = 1

def run_single(output_name, ind = 0, mean=0.85, std=0.25, nprop=nprop_default, sprop=sprop_default, omega=1.5, email_key=None):
  print('---------------------------------------------------------------------------')
  print(f'Run {run_count} : {output_name}')
  print('---------------------------------------------------------------------------')
  mem_tags = np.random.randn(n, n)
  mem_tags = gram_schmidt_process(mem_tags)
  
  np.save("mem_tags.npy", mem_tags)
  print(f"Random generated memory tags:\n{mem_tags}")
  print(f"Memory tags shape: {mem_tags.shape}")
  mem_components = encode_images(store_img_paths, mem_tags[:-1], [mean] * (n-1), [std] * (n-1))
  
  print(f"Encoded memory components shape: {mem_components.shape}")
  
  nn = MemorySNN(  # NN 구축
      N=mem_components.shape[-1],
      neuron_prop=nprop,
      synapse_prop=sprop
  )
  
  inputgen = SineWaveInputGenerator(mem_components, omega)
  
  print("Learning memory components...")
  nn.learn_memory(inputgen, steps=200, dt=0.01)
  
  # box cue signal 준비

  mem_components = encode_box_images(store_img_paths, mem_tags[:-1], mean, std, index=ind)
  relevant_noised_cue = mem_components[0]
  
  print("Relevant noised cue signal:")
  decode_neural_state(relevant_noised_cue, mem_tags[:-1])

  
  plt.show()
  plt.savefig('result/noised_signal.png')
  plt.close()
  
  nn.clear()
  print("Retrieving images from the cue signal...")
  imgs = encode_images_no_tag(store_img_paths, [mean] * (n-1), [std] * (n-1))
  nn.retrieve_from_cue(np.stack([relevant_noised_cue] * 5), mem_tags[:-1], omega, [imgs[i].reshape(32, 32) for i in range(5)])

  if email_key is not None:
    send_email_with_zip(
        zip_path=zip_directory('result', zip_name=output_name+'.zip'),
        subject="작업 파일 압축본",
        body="첨부된 ZIP 파일을 확인하세요.",
        sender="resultsender2025@gmail.com",
        receiver="24065@sshs.hs.kr, 24072@sshs.hs.kr, 24046@sshs.hs.kr",
        smtp_server="smtp.gmail.com",
        smtp_port=465,
        password=email_key  # Gmail 앱 비밀번호 사용
    )
