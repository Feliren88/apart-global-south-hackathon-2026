https://huggingface.co/datasets/akanshjain37/counterfactual-pendulum-multilingual/

https://huggingface.co/datasets/feliren/multilingual-counterfactual

https://huggingface.co/datasets/Anvesh-Lankala/remote_sensing_VQA_multilingual

https://huggingface.co/datasets/Anvesh-Lankala/multilingual-crossmodal-conflict-3D_Objects

you are principal applied scientist in AI safety from Antrophic. you want to do research on counterfactual prompt and its effect towards multiple VLMs accross small and mid size like 20B one from open weights one from hugging face.


I have above datasets. I want to develop a python script in @/home/vfeliren1/lf93_scratch2/vfvic1/apart-global-south-hackathon-2026/inference where it does
1. load those datasets and models. models should be VLM from hugging face and user should be able to try it out multiple models
2. Do prompting based on the counter factual (counterfactual_caption) one and the image so it should be VLM
3. Add mcq_question in there to get the answer for VLM 
4. Check whether VLM models are leaning towards text_bias or image_bias or plausible_distractor or outside those 3
5. Try all languages and see what happend
6. For each output, save all of them including all configs, languages, activation internal mechanism hooks, answer and the counter for each image_answer_bias, text_answer_bias, plausible_distractor or others. Ensure every informations are worth for analysis towards research papers especially in AI safety, mechanistic interpretability, etc
7. make bash script where I can make config on language, all languages, which dataset to try, what models to try, and more other config. 

for models, you can try
https://huggingface.co/sarvamai/sarvam-30b -- This is the 30 B model from India
https://huggingface.co/collections/Sahabat-AI/sahabat-ai-v1 sahabat-ai from Indonesia
Aya-Vision-8b, Qwen3-VL-8B, Intern-VL3-8B,Qwen2.5-VL-7B,LLaVa-Onevision-7B and also more params and ensure those are in config. 
