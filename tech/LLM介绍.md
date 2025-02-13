# LLM介绍
(本文内容主要来自Andej Karpathy的入门视频。Karpathy是业内知名的人工智能大神，师从李飞飞，是OpenAI的创始员工之一，曾任特斯拉人工智能负责人。)
本文主要介绍了训练大模型的三大步骤：预训练、后训练和强化学习，以及每一步的方法和目的是什么。最后，还单独介绍了deepseek的技术创新点。新人如我，看完本文应该会对大语言模型有一个基本够用的了解了。
## 预训练
### 预训练数据来源
预训练是一种模型训练的策略，通常在大规模的数据集上进行。在这个过程中，模型会先使用大量未标注（或少量标注）的数据进行初步训练，目的是学习数据的通用特征表示或知识。预训练的核心目标是通过在大规模数据集上的训练，使模型学习到数据的通用特征和规律。这些特征具有泛化能力，可以应用于多个不同的任务和领域。预训练完成后，模型通常会作为一个基础模型，用于后续的微调（Fine-tuning）或直接应用于相关任务。

所以预训练的最重要诉求就是训练数据要大、要杂、要通用好泛化。于是，我们最先需要考虑的是用什么数据来进行预训练。没有数据，一切都白搭。自然而然的，我们想到了互联网。互联网上每天都在产生成千上万的新内容，积累更是天文数字。考虑到搜索引擎的爬虫机制，是否技术上存在一种可能性，即用爬虫把全互联网的数据都给爬下来当作数据源呢？进一步想想，搜索引擎公司是不是已经拥有了很多数据积累了呢？再进一步想想，除了搜索引擎以外的其他互联网巨头，比如社交、电商，是不是也都已经积累了海量的数据呢？这个逻辑也解释了这样的现象：即国外最早大投入搞AI的是google，而国内是百度。随后，互联网巨头们纷纷下下场。而且我们听到了大量的关于数据才是本质的说法，国家政策层面也更为重视数据而不是模型本身。

对于原始的互联网数据，还需要进行大量的预处理。想象一下，对于一个网页，进行训练需要收集只是页面内容。但我们如果看一下源码，里面必然存在着各种tag，这些是需要清洗掉的。除此以外，还有很多事情要做，举几个例子：
- 比如对于特定的url进行过滤，比如来自不可靠的甚至是钓鱼网站的数据。
- 个人敏感信息(PII)的过滤处理。
- 内容质量的过滤器
- 多语言的处理
- 自定义的各种过滤器
很多互联网巨头在平日的数据积累过程中已经部分或全部的处理了这些步骤了。于是他们具备先发优势也是合理的。

但上述逻辑出现了一个问题，即如果数据的数据和预处理如此重要，那大型互联网公司理应垄断AI，但为什么很多技术进步反而出现在小公司呢，比如OpenAI或者Deekseek？答案是开源。这对于其他行业来说也许不常见，但开源共享的精神确实是几十年来不断推动互联网飞速发展的重要因素之一。
打开https://huggingface.co/datasets, 我们能看到大量的开源数据集：
- 这里既有可以用于预训练的通用超大型数据集，例如https://huggingface.co/datasets/HuggingFaceFW/fineweb  
- 也有后文会提到的用于后训练的数据集，例如 https://huggingface.co/datasets/deepseek-ai/DeepSeek-Prover-V1

fineweb是个英文的数据集，从2013年，每个月的数据收集在一起。目前一个月的数据大约在600G+大小，相当于2000亿+个token。累积起来的那就是tb级别了。
格式参考如下：

{
   "text": "This is basically a peanut flavoured cream thickened with egg yolks and then set into a ramekin on top of some jam. Tony, one of the Wedgwood chefs, suggested sprinkling on some toasted crushed peanuts at the end to create extra crunch, which I thought was a great idea. The result is excellent.",
   "id": "<urn:uuid:e5a3e79a-13d4-4147-a26e-167536fcac5d>",
   "dump": "CC-MAIN-2021-43",
   "url": "<http://allrecipes.co.uk/recipe/24758/peanut-butter-and-jam-creme-brulee.aspx?o_is=SimilarRecipes&o_ln=SimRecipes_Photo_7>",
   "date": "2021-10-15T21:20:12Z",
   "file_path": "s3://commoncrawl/crawl-data/CC-MAIN-2021-43/segments/1634323583083.92/warc/CC-MAIN-20211015192439-20211015222439-00600.warc.gz",
   "language": "en",
   "language_score": 0.948729,
   "token_count": 69
}

秉持着开源精神，大部分数据集都是已经整理归纳好的而且都是开源协议的。所以，直接拿着用就行了。相比于数据本身，预训练的门槛其实来到了算力。
### 标记化(tokenization)

raw data -> symbols
### 神经网络的输入输出
sequence of tokens called as network input
probabilities of token shows next as network output
use weight for tokens to adjust network
### 结果联想
since weights are fixed, the model is fixed.
every question to model is not training any more.
the answer from model is inference.
### GPT-2
1.5b, max 1024 token, trained on 100b tokens.

## 后训练
human lablers, ideal q&a into model. new conversation dataset
modify into "<im_start>...<im_sep>...<im_end>"

https://arxiv.org/pdf/2203.02155
gpt没开源，openassistant , ultrachat
chatgpt的返回的其实是对人类作者的答案的一种统计学意义上的模拟

### 幻觉的产生及可能的解决方式
token sample, not token 
the llama3 herd of models --- add question that answer is "i don't know".
1. 选择权威文件，然后构造对应的问题。然后向llm提问。如果llm不懂，就构造一个特定的问答是“i dont' know".
2. allow to search <search_start>...<search_end>
3. attach context for llm
4. 说自己是chatgpt，也有可能是幻觉。需要输入特定问答才行。hardcode it.

### token训练逻辑的迷思
token从左向右，所以有逻辑的答案才是好的结构。一上来给出结果会导致训练效果很差。
强制要求压缩推理过程可能导致错误的结果(gpt o4),有点像心算。需要一步步的，结果才更准确。
不太擅长数数。用代码实现，简化了对于model的负担。
主要是因为token可能不会按照人类的意愿划分，model是按照token处理问题的，没有整体性。
9.11>9.9?

## 强化学习
预训练，书本中的知识介绍
sft，书中的问题及解答
rl,书本中的问题及答案，自己找寻解答。
对于给定的问题和最终答案，让模型自己尝试解题步骤，并把答案对的继续用来反复训练，一遍又一遍。重复很多遍。
rl的训练量太大，使用rlhf来提升效率。但全用人力成本又太高，所以建立一个reward model, 然后进行rl训练， 用reward model替代真人介入。

## deep seek
deepseek还是rl，但是有了aha moment.相当于rl的解答到一半时，回退。会用大量的token，但是准确率大幅提高。
chatgpt o3是类似的推理模型，但是没有显示推理效果。

R1 是基于预训练好的 DeepSeekV3 基础模型进行的后训练优化，V3 做了很多创新优化，很大降低了预训练模型和模型推理的成本，是 R1 的基础。
R1-Zero 证明了对已预训练好的模型，不需要经过 SFT，只需要纯粹的 RL，就能让模型涌现 CoT 推理能力。sft成本高，rl简单但计算量大。

R1-Zero 最大的不同，是在强化学习中使用 GRPO 算法代替 PPO。每次输出一组结果，并且比较。
奖励模型基于规则,判断答案并判断格式，计算简单，准确性好。让模型使用工具或者搜索来提升对于结果的判断。

结果就是推理步数越来越长，效果越来越好。出现了aha时刻。

R1再重复r1-zero之前，进行了超高质量的sft，让输出对于人类变的友好。再用r1生成的60w推理数据+20w文学数据进行新一轮sft，最后进行一轮简单的强化学习矫正。

蒸馏，用80万数据对于其他小模型进行sft，效果贼好。
