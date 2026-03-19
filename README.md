# nanoGPT
⽬前市⾯上⼤多数可⽤的GPT都⾮常的庞⼤，初学者很难学习和复现。nanoGPT项⽬[1]是使⽤Pytorch对GPT的⼀个复现，
包括训练和推理，其⽬的是做到⼩巧、⼲净、可解释性并且能⽤于教育。如果市⾯上的GPT模型是⼀艘“航空⺟ 舰”的话，
nanoGPT可看做是⼀艘游艇，“⿇雀虽⼩，五张俱全”，对于初学者的⼊⻔学习有重要的意义。
本案例将训练两个模型：⼀个是使⽤由58000⾸诗词构成的诗歌数据集，
训练⼀个歌词⽣成的GPT；另⼀个是使⽤约 124万个字符构成的《天⻰⼋部》⽂本，训练⼀个具有《天⻰⼋部》⻛格的GPT。

# 复现实验nanoGPT实验
## 安装配置库
`pip install torch numpy transformers datasets tiktoken wandb tqdm`
## 生成数据集
运行`python data/shakespeare_char/prepare.py`  
生成数据集莎士比亚input.txt文件
![img.png](markdown图片/img.png)
## 训练数据集
运行`python train.py config/train_shakespeare_char.py`  
其训练过程截图：  
![img.png](markdown图片/图片2.png)
## 运行效果
运行`python sample.py --out_dir=out-shakespeare-char`  
通过采样脚本进行抽取结果  
其结果（部分）：
```
Overriding: out_dir = out-shakespeare-char
number of parameters: 10.65M
Loading meta from data\shakespeare_char\meta.pkl...


Clown:
So, who? and it shall bear the way
Of the grave that I am supposed foul and very redeems.

Shepherd:
What's a wizard with the tofging come and thing,
We might send in this maid over-banished prepared;
And well in a wallow up, on him so shrink.

Shepherd:
But I will attorney him.

Clown:
He knocks not what evils I have done with him;
But look your highness dangerous for a sight!

Shepherd:
Menenius, a Prince, Sir, and Norfolk,
Should bad all his grace hither to a rage
His hands and for hi
---------------

Men pardon me, you shall have hang you; I see
your father than the shepherd: he will but be so gone.

Shepherd:
Here is.

AUTOLYCUS:
He was a servant when it stands it.

LARTIUS:
My liege.

AUFIDIUS:
Why, my lord?

Clown:
I have no more, if I have so many have been with
him to me way for the vield of Corioli large.

AUTOLYCUS:
I say it is not to be not.

CORIOLANUS:
But to this often nor the sun of Rome, nor what thou wilt say
this belly a punishment and back watch-time to the sea?
Good knee you
---------------

Men than I beseech your husband;
And yet I said, I see this double council or love
As yourselves, sir; tell me, first, I will tell you.

BUCKINGHAM:
I will be cause your age the crown.

KING RICHARD III:
Good lord, brave sir, but madam:
That yet I will not speak a fool, and have left me so.
The old conscience did some crown of him;
And back your shin; you shall be thus a quarrel hate your grace.

KING RICHARD II:
The babe where you hear, no tongue.

KING RICHARD II:
First, I am not a little son

---------------
```

# 
