大家好，我是煎鱼。

最近 ChatGPT 很火，AIGC 很火，各类国产化 AI 很火。周边的 AI 工具集、框架也很火。各类新词也层出不穷。

今天和大家学习和分享的是重量级新选手 LangChain。

## 什么是 LangChain

LangChain 是一个 2023 年 1 月（v0.0.64）在 GitHub 上新开源的新框架，框架的作用是可以通过可组合性使用 LLM 构建你的应用程序。

现阶段更新频率较高。有 Python 和 JS 的两种版本。和 AIGC 一样的热度，广受追捧，Stars 已经冲到了 38k 左右。

![](https://files.mdnice.com/user/3610/26e40457-5f76-4e58-b343-d66f9f7d2936.png)

官方的说辞：LangChain 是一个用于开发由语言模型驱动的应用程序的框架。我们相信，最强大、最与众不同的应用程序不仅会通过API调用语言模型，而且还会：
- 具有数据意识：将语言模型与其他数据源连接起来。
- 具有自主性：允许语言模型与其环境进行交互。

总的而言，LangChain 设计目标就是支持以上这些类型的应用程序。

不知名网友的说辞：看起来是个基于语言模型（LLM）各路集成、封装，串联、扩展，提供开箱即用的框架集，看起来概念不少。

## 价值在哪

LangChain 说，自己的主要价值在于以下：
- 组件化（Components）：LangChain 为使用语言模型所需的组件提供了模块化的抽象。LangChain 还为所有这些抽象提供了实现的集合。
- 特定于用例的链（Use-Case Specific Chains）：链可以被认为是以特定的方式组装这些组件，以便最好地完成特定的用例。它们旨在成为一个更高级的接口，人们可以通过它轻松地开始使用特定的用例。这些链也被设计为可定制的。

第一点：很好理解，LangChain 把各种东西做成了组件化、模块化，我们可以直接填个 OpenAI Key 就实现开箱即用，或是自己实现这个组件。就是做了各类的抽象接口化。

第二点：看着有点绕口。一开始听别人介绍 LangChain 时，也总满口链什么什么的。。。一开始真的不太理解。其实照官方的信息一看，这就是支持场景化的意思，可以把一个个行为拼装成一个 ”链“。

具体的组件和链会在后面再进一步展开。

## 安装和使用

我们先安装，直接用 pip 或 pip3 都可以。在命令行运行如下安装命令。

安装 langchain：

```shell
$ pip3 install langchain
```

安装 openai：

```shell
$ pip3 install openai
```

安装完后，这两个软件库就已经被拉取到你的本地目录下了。

如果有兴趣查看，可以使用：`pip3 show -f openai` 命令，就看到 openai 这个库的具体信息和目录：

```
$ pip3 show -f openai
Name: openai
Version: 0.27.6
Summary: Python client library for the OpenAI API
Home-page: https://github.com/openai/openai-python
Author: OpenAI
Author-email: support@openai.com
License: None
Location: /usr/local/lib/python3.9/site-packages
Requires: tqdm, aiohttp, requests
Required-by: 
Files:
  ../../../bin/openai
  openai-0.27.6.dist-info/INSTALLER
  openai-0.27.6.dist-info/LICENSE
```

## 快速 Demo

安装完毕后，快速写一个 LangChain Demo。在你常用的代码目录，新建一个 .py 文件，用于编写 Demo 代码。

代码如下：

```python
import os
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "设置你的 OpenAI KEY"

llm = OpenAI(temperature=0.9)

text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
```

输出结果：

```
Rainbow Sockz
```

每次输出的结果可能都不太一样。作为一个 Demo，我们先跑起来，后续再进一步深究。

注意：如果调用不通的话，可能需要科学上网。

## 使用 OpenAI KEY

Demo 代码中的 OpenAI 的 KEY 可以到 `https://platform.openai.com/account/api-keys` 获取或创建一个新的。

![](https://files.mdnice.com/user/3610/61daa218-10a6-4eb7-bff0-b17808770e0f.png)

在 Usage 栏目也可以看到自己当前账号的使用额度和调用情况。早期的账号是有默认送 18 美金的免费额度，现在（2023年中）的是送 5 美金。

如下图：

![](https://files.mdnice.com/user/3610/83b7d6bf-432f-406e-9f16-e0c863cbcd1f.png)

## 总结

今天我们快速的介绍了 LangChain 是一个基于 LLM 的各路集成、封装，串联、扩展，提供开箱即用的框架集，主要价值是组件和链。初步的涉猎了里面的一些大概念。

紧接着我们运行了一个 Demo，初步跑通了 LangChain。完成了第一步的学习和了解。后续我们会继续深入。
