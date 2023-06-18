大家好，我是煎鱼。

ChatGPT 的爆火，莫过于在聊天交互上的神化。一时间各大业务厂家纷纷整起了新活。这其中相对综合成本较低的，就是基于知识库的问答 AI 机器人。

今天我们将基于 LangChain 快速实现一个 AI 客服，让我们在日常工作都能用上。

除了之前章节中安装的 python 第三方依赖包外。过程中我们至少还需要安装如下依赖：

```python
$ pip install unstructured
$ pip install chromadb
$ pip install tiktoken
```

首先我们在项目目录下，新建 testdata 目录，并创建 data.txt 文件。写入一些你计划用于问答的知识库内容。

我这里是 cv 了一些腾讯维基百科的资料：

```
腾讯控股有限公司（英语：Tencent Holdings Limited），简称腾讯，是中国一家跨国企业控股公司，为中国大陆规模最大的互联网公司，1998年11月由马化腾、张志东、陈一丹、许晨晔、曾李青5位创始人共同创立，总部位于深圳南山区腾讯滨海大厦。腾讯业务拓展至社交、金融、投资、资讯、工具和平台等不同领域，其子公司专门从事各种全球互联网相关服务和产品、娱乐、人工智能和技术。目前，腾讯拥有中国大陆使用人数最多的社交软件腾讯QQ和微信，以及最大的网络游戏社区腾讯游戏。在电子书领域 ，旗下有阅文集团，运营有QQ阅读和微信读书。

腾讯于2004年6月16日在香港交易所挂牌上市，于2016年9月5日首次成为亚洲市值最高的上市公司[1]，并于2017年11月21日成为亚洲首家市值突破5000亿美元的公司[2]。2017年，腾讯首次跻身《财富》杂志世界500强排行榜，以228.7亿美元的营收位居478位[3]。

香港财经界把阿里巴巴、腾讯、美团点评、小米四只中国大陆科技股的英文名称首个字母，合称“ATMX”股份。[4]

腾讯QQ，原名OICQ，是腾讯公司创始人马化腾于1999年开发的即时通讯软件。因与另一款1996年11月开发的即时通讯软件ICQ（目前ICQ已经归AOL旗下所有，腾讯间接控股）名字相似而被控侵权。虽然腾讯最终胜诉，但仍然将 OICQ 更名为 QQ[41]。

微信是腾讯旗下针对智能手机平台开发的带有网络社交功能的即时通讯应用，后来拓展到多样化的生活平台。由张小龙担任部门负责人。
```

在具体的业务逻辑上，分为以下几个步骤：

1、加载知识库资料，使用之前提到的文档加载器。在本文这个案例中，用的是 txt 文件。

```python
loader = DirectoryLoader('./testdata/', glob='**/*.txt')
documents = loader.load()
```

2、加载完毕后，对加载的文本进行长度切割。也就是我们常说的段落切割后分块。

```python
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)
```

3、初始化分块文本的向量数据，并存进向量数据库。便于后续使用余弦进行匹配。

```python
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(split_docs, embeddings)
```

4、创建 LangChain 的问答对象，会完成一系列 OpenAI 调用和业务逻辑处理。内部主要是问题的向量化后到向量数据库进行余弦匹配，然后筛选出 TopN，再具体你的配置情况，选择合适的问题答案。

```python
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    )

# 进行问答
rsp = qa({"query": "QQ原名是什么？"})
print(rsp)
```

分块的业务逻辑梳理如上。

完整的程序代码如下：

```python
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain import OpenAI, VectorDBQA
import os

os.environ["OPENAI_API_KEY"] = "xxxx"


# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader('./testdata/', glob='**/*.txt')
documents = loader.load()

# 切割文本长度，进行分块
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

# 初始化 openai 的 embeddings 对象，使用 Chroma 向量数据库
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(split_docs, embeddings)

retriever = docsearch.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["k"] = 4

# 创建问答对象
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    )

# 进行问答
rsp = qa({"query": "QQ原名是什么？"})
print(rsp)
```

运行结果：

```
$ python main.py
Created a chunk of size 266, which is longer than the specified 100
Created a chunk of size 140, which is longer than the specified 100
Created a chunk of size 128, which is longer than the specified 100
{'query': 'QQ原名是什么？', 'result': ' OICQ.', 'source_documents': [Document(page_content='腾讯QQ，原名OICQ，是腾讯公司创始人马化腾于1999年开发的即时通讯软件。因与另一款1996年11月开发的即时通讯软件ICQ（目前ICQ已经归AOL旗下所有，腾讯间接控股）名字相似而被控侵权。虽然腾讯最终胜诉，但仍然将 OICQ 更名为 QQ[41]。', metadata={'source': 'testdata/data.txt'}), Document(page_content='微信是腾讯旗下针对智能手机平台开发的带有网络社交功能的即时通讯应用，后来拓展到多样化的生活平台。由张小龙担任部门负责人。', metadata={'source': 'testdata/data.txt'}), Document(page_content='腾讯控股有限公司（英语：Tencent Holdings Limited），简称腾讯，是中国一家跨国企业控股公司，为中国大陆规模最大的互联网公司，1998年11月由马化腾、张志东、陈一丹、许晨晔、曾李青5位创始人共同创立，总部位于深圳南山区腾讯滨海大厦。腾讯业务拓展至社交、金融、投资、资讯、工具和平台等不同领域，其子公司专门从事各种全球互联网相关服务和产品、娱乐、人工智能和技术。目前，腾讯拥有中国大陆使用人数最多的社交软件腾讯QQ和微信，以及最大的网络游戏社区腾讯游戏。在电子书领域 ，旗下有阅文集团，运营有QQ阅读和微信读书。', metadata={'source': 'testdata/data.txt'}), Document(page_content='腾讯于2004年6月16日在香港交易所挂牌上市，于2016年9月5日首次成为亚洲市值最高的上市公司[1]，并于2017年11月21日成为亚洲首家市值突破5000亿美元的公司[2]。2017年，腾讯首次跻身《财富》杂志世界500强排行榜，以228.7亿美元的营收位居478位[3]。', metadata={'source': 'testdata/data.txt'})]}
```

顺利运行，答案也是正确的。

这就是一个最简版的基于知识库的 ChatQA 了，快尝试跑起来吧！

后续我们会继续深入，看看有没有什么更好玩的黑科技功能和配置。
