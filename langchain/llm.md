大家好，我是煎鱼。

在前面的快速入门中，我们提到了 LangChain 的两大价值：模块（Modules）和特定于用例的链（Use-Case Specific Chains），这两大块把整个 LangChain 给支棱了起来。

## 有什么模块

模块，听起来就像一个各功能组件的大集合，什么都可以包含在内的感觉。确实。就是，在 LangChain 的官方文档中，链也包含在模块里。

模块共包含如下几类：
- 模型（Models）
- 提示（Prompts）
- Indexes
  - 文档加载器（Document Loaders）
  - 文本切割器（Text Splitters）
  - 检索器（Retriever）
  - 向量商店（Vectorstore）
- 内存（Memory）
- 链（Chains）
- 代理（Agents）
- 回调（Callbacks）

概念之多，我们将会分开来讲解和实操。

## 模型

在本文中我们先要介绍的是模型（Models），这是我们使用所有库、工具，写应用的基石。不跑通，都运行不起来。

共涉及如下几类：
- 大型语言模型（LLMs）
- 聊天模型（Chat Models）
- 文本嵌入模型（Text Embedding Models）

### 大型语言模型（LLMs）

LangChain 不是 OpenAI，本身不提供 LLM。他在所有模块设计中有着标准化接口的理念。这和 Go 的鸭子模型挺相近。你可以通过实现接口，以满足与各式各样的 LLM 实现接口对接。

#### 快速 Demo

在 LangChain 中使用 LLM 非常简单。我们只需要按如下代码使用：

```python
import os
from langchain.llms import OpenAI

// 需注意设置你的 OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "xxx"

llm = OpenAI(model_name="text-davinci-003",max_tokens=1024)
print(llm("Go 语言是什么？"))
```

此处设置了 LLM 使用的 model 是 text-davinci-003，最大使用的 token 数是 1024，避免产生过多的不必要费用。

运行结果：

```
Go（又称 Golang）是由 Google 开发的一种静态强类型、编译型、并发型，并具有垃圾回收功能的编程语言。它是一种可以轻松构建快速、可伸缩的网络服务和系统软件的语言。Go 语言具有高效的编译速度，便携性，动态链接，以及简单的语法。
```

当然，测试代码一般直接这么运行就可以：

```

llm = OpenAI(temperature=0.9)

print(llm.predict("Go 语言有机会打赢 Java 语言吗？"))
```

直接调用 predict 方法，该方法的作用的定义是 “预测”。

### 聊天模型（Chat Models）

聊天模型的使用，是目前最火的一个领域。最初 ChatGPT 的登峰造极，就是由大家在和聊天 GPT 玩耍的过程中登上的热议。

我们可以设置提示语（Prompt），并进行单轮或多轮对话作为输入，GPT 会为我们返回流式或非流式的响应结果。

#### 快速 Demo

我们先快速运行一个 Demo，如下代码：

```go
// 需注意默认设置你的 OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "xxx"

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
s = chat.predict_messages([HumanMessage(content="请写一个 Go 语言的 Hello world 给我，希望可以高级一些！")])
print(s.content)
```

AIMessage、HumanMessage、SystemMessage 分别指代不同的角色。一般常用的是 HumanMessage。其余的在后续设定和多轮对话等场景会用到。

输出结果：

```go
package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello, World!")
}
```

这是一个简单的 Go 语言程序，它输出 "Hello, World!" 到控制台。使用 fmt 包中的 Println 函数来输出文本。在 main 函数中调用该函数。（这段也是 GPT 一起给出的程序解释）

## 应用尝鲜

在了解了如何调用后。我们进一步学习一个完整的 ChatGPT 应用如何调用。

这里主要涉及到两个概念，一个是提示语（Prompt）和链（Chains）。前者是 OpenAI 本来就有的，是一种 NLP 中的概念，用于指导机器学习模型生成特定类型、主题或格式等输出。

后者是 LangChain 的应用概念。链就是将多个组件组合串联运行的作用，例如：模型、提示和其他链。是 LangChain 必学的一个组成只是。

如下代码：

```python
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)
template = "你是一个资深的编程语言高手，请帮我把这段 {input_language} 程序翻译成 {output_language} 程序。"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

txt = '''

package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello, World!")
}

'''

chain = LLMChain(llm=chat, prompt=chat_prompt)
s = chain.run(input_language="Go", output_language="Python", text=txt)
print(s)
```

请关注 SystemMessagePromptTemplate、HumanMessagePromptTemplate 及其对应使用的 template、text 变量等。

虽然看起来代码不少，但实际上就是字符串模板的拼接和赋值。最后通过 LLMChain 将 LLM Model 和 Prompt 组合起来，再发起变量赋值和调用。最后输出结果。

输出结果：

```python
# Python equivalent of the Go program

print("Hello, World!")
```

## 总结

今天我们介绍了 LangChain 中比较常用的组件有哪些，其中我们挑选了大家最感兴趣的 LLMs 和 Chat Models 进行了应用的演示。

其中我们还涉及到了 Prompt 和 Chains 的使用，这是非常关键的概念。大家务必学习和亲自运行一遍。
