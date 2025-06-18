# 필요한 라이브러리 불러오기
from snnlib import *
import os, sys

# zip파일로 묶을 폴더 준비
try:
  os.mkdir('result')
except:
  pass

# 1번 그래픽카드 사용(필요에 따라 변경)
cp.cuda.Device(1).use()

# 이미지 경로 불러오기
img_paths = glob("images/*.png")
store_img_paths = [p for p in img_paths if "forest" not in p]  # forest 사진 제외 악기 사진 5개 사용
store_img_paths.sort()

n = len(img_paths)
print(f"Number of images: {n}")

# 뉴런 하이퍼파라미터 기본값
nprop_default = NeuronProp(V_rest=0.,
                           V_th=0.06,
                           tau_m=0.02,
                           E_ex=0.06,
                           tau_e=0.02)

# 시냅스 하이퍼파라미터 기본값
sprop_default = SynapseProp(tau_pre=0.02,
                            tau_post=0.02,
                            dA_pre=0.01,
                            dA_post= -0.0105,
                            g_max=10.,
                            w_ratio=1/310)

# 실행한 총 횟수 저장
run_count = 1

# 한 번의 학습 및 테스트 사이클 함수
def run_single(output_name, ind = 0, mean = [0.9]*5, std=0.25, nprop=nprop_default, sprop=sprop_default, omega=1.5, email_key=None):
  print('---------------------------------------------------------------------------')
  print(f'Run {run_count} : {output_name}')
  print('---------------------------------------------------------------------------')
  # 직교 벡터 생성
  mem_tags = np.random.randn(n, n)
  mem_tags = gram_schmidt_process(mem_tags)

  # 이미지 인코딩
  np.save("mem_tags.npy", mem_tags)
  print(f"Random generated memory tags:\n{mem_tags}")
  print(f"Memory tags shape: {mem_tags.shape}")
  mem_components = encode_images(store_img_paths, mem_tags[:-1], mean, [std] * (n-1))
  
  print(f"Encoded memory components shape: {mem_components.shape}")
  
  nn = MemorySNN(  # NN 구축
      N=mem_components.shape[-1],
      neuron_prop=nprop,
      synapse_prop=sprop
  )

  # 입력 정의
  inputgen = SineWaveInputGenerator(mem_components, omega)

  # 학습
  print("Learning memory components...")
  nn.learn_memory(inputgen, steps=200, dt=0.01)
  
  # box cue signal 준비
  mem_components = encode_box_images(store_img_paths, mem_tags[:-1], mean[0], std, index=ind)
  relevant_noised_cue = np.array([mem_components[0]]*5)
  
  print("Relevant noised cue signal:")
  decode_image(relevant_noised_cue, mem_tags[:-1])

  # box cue signal 저장
  plt.show()
  plt.savefig('result/noised_signal.png')
  plt.close()

  # 테스트
  nn.clear()
  print("Retrieving images from the cue signal...")
  imgs = encode_images_no_tag(store_img_paths, mean, [std] * (n-1))
  nn.retrieve_from_cue(relevant_noised_cue , mem_tags[:-1], omega, [imgs[i].reshape(32, 32) for i in range(5)])

  # gmail로 이메일 보내기
  if email_key is not None:
    send_email_with_zip(
        zip_path=zip_directory('result', zip_name=output_name+'.zip'),
        subject="작업 파일 압축본",
        body="첨부된 ZIP 파일을 확인하세요.",
        sender="resultsender2025@gmail.com", # 실험자 개인 이메일 전송용 이메일
        receiver="24065@sshs.hs.kr, 24072@sshs.hs.kr, 24046@sshs.hs.kr", # zip파일을 수령할 실험자들
        smtp_server="smtp.gmail.com",
        smtp_port=465,
        password=email_key  # Gmail 앱 비밀번호 사용
    )
