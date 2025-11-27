let currentSampleType = "text"
let generatedSamples = []
let currentAdvantageSample = null

// Select sample type
function selectSampleType(type) {
  currentSampleType = type

  // Update active card
  document.querySelectorAll(".sample-type-card").forEach((card) => {
    card.classList.remove("active")
  })
  event.target.closest(".sample-type-card").classList.add("active")

  // Update configuration panel
  updateConfigPanel(type)
}

// Update configuration panel based on sample type
function updateConfigPanel(type) {
  const configContent = document.getElementById("configContent")

  switch (type) {
    case "text":
      configContent.innerHTML = `
                <div class="config-group">
                    <label class="config-label">文本类型</label>
                    <select class="config-input" id="textType">
                        <option value="article">文章</option>
                        <option value="review">评论</option>
                        <option value="description">描述</option>
                        <option value="story">故事</option>
                        <option value="news">新闻</option>
                    </select>
                </div>
                <div class="config-group">
                    <label class="config-label">主题关键词</label>
                    <input type="text" class="config-input" id="keywords" placeholder="科技, 人工智能, 创新">
                </div>
                <div class="config-group">
                    <label class="config-label">文本长度</label>
                    <select class="config-input" id="textLength">
                        <option value="short">短文本 (50-150字)</option>
                        <option value="medium" selected>中等长度 (150-300字)</option>
                        <option value="long">长文本 (300-500字)</option>
                    </select>
                </div>
                <div class="config-group">
                    <label class="config-label">语言风格</label>
                    <select class="config-input" id="textStyle">
                        <option value="formal">正式</option>
                        <option value="casual" selected>随意</option>
                        <option value="professional">专业</option>
                        <option value="creative">创意</option>
                    </select>
                </div>
                <div class="config-group" style="grid-column: 1 / -1;">
                    <label class="config-label">额外要求</label>
                    <textarea class="config-input config-textarea" id="additionalRequirements" placeholder="请描述任何特殊要求或约束条件..."></textarea>
                </div>
            `
      break

    case "data":
      configContent.innerHTML = `
                <div class="config-group">
                    <label class="config-label">数据类型</label>
                    <select class="config-input" id="dataType">
                        <option value="user">用户信息</option>
                        <option value="product">产品数据</option>
                        <option value="transaction">交易记录</option>
                        <option value="log">日志数据</option>
                    </select>
                </div>
                <div class="config-group">
                    <label class="config-label">字段配置</label>
                    <input type="text" class="config-input" id="fields" placeholder="name, age, email, phone">
                </div>
                <div class="config-group">
                    <label class="config-label">数据范围</label>
                    <input type="text" class="config-input" id="dataRange" placeholder="年龄: 18-65, 价格: 10-1000">
                </div>
                <div class="config-group">
                    <label class="config-label">数据格式</label>
                    <select class="config-input" id="dataFormat">
                        <option value="realistic">真实化</option>
                        <option value="random">随机化</option>
                        <option value="pattern">规律化</option>
                    </select>
                </div>
            `
      break

    case "code":
      configContent.innerHTML = `
                <div class="config-group">
                    <label class="config-label">编程语言</label>
                    <select class="config-input" id="language">
                        <option value="javascript">JavaScript</option>
                        <option value="python">Python</option>
                        <option value="java">Java</option>
                        <option value="html">HTML/CSS</option>
                        <option value="sql">SQL</option>
                    </select>
                </div>
                <div class="config-group">
                    <label class="config-label">代码类型</label>
                    <select class="config-input" id="codeType">
                        <option value="function">函数</option>
                        <option value="class">类</option>
                        <option value="algorithm">算法</option>
                        <option value="component">组件</option>
                    </select>
                </div>
                <div class="config-group">
                    <label class="config-label">复杂度</label>
                    <select class="config-input" id="complexity">
                        <option value="simple">简单</option>
                        <option value="medium" selected>中等</option>
                        <option value="complex">复杂</option>
                    </select>
                </div>
                <div class="config-group">
                    <label class="config-label">功能描述</label>
                    <input type="text" class="config-input" id="functionality" placeholder="排序算法, 数据处理, API接口">
                </div>
            `
      break

    case "conversation":
      configContent.innerHTML = `
                <div class="config-group">
                    <label class="config-label">对话场景</label>
                    <select class="config-input" id="scenario">
                        <option value="customer_service">客服对话</option>
                        <option value="interview">面试对话</option>
                        <option value="casual">日常聊天</option>
                        <option value="business">商务沟通</option>
                    </select>
                </div>
                <div class="config-group">
                    <label class="config-label">参与人数</label>
                    <select class="config-input" id="participants">
                        <option value="2">双人对话</option>
                        <option value="3">三人对话</option>
                        <option value="group">群组对话</option>
                    </select>
                </div>
                <div class="config-group">
                    <label class="config-label">对话长度</label>
                    <select class="config-input" id="conversationLength">
                        <option value="short">短对话 (5-10轮)</option>
                        <option value="medium" selected>中等长度 (10-20轮)</option>
                        <option value="long">长对话 (20-30轮)</option>
                    </select>
                </div>
                <div class="config-group">
                    <label class="config-label">对话主题</label>
                    <input type="text" class="config-input" id="conversationTopic" placeholder="产品咨询, 技术讨论, 日常生活">
                </div>
            `
      break
  }
}

// Generate samples
function generateSamples() {
  const count = Number.parseInt(document.getElementById("sampleCount").value)
  const trainingCorpus = document.getElementById("trainingCorpus").value
  const selfEvalPrompt = document.getElementById("selfEvaluationPrompt").value
  const temperature = Number.parseFloat(document.getElementById("temperature").value) || 0.7
  const topK = Number.parseInt(document.getElementById("topK").value) || 50
  const topP = Number.parseFloat(document.getElementById("topP").value) || 0.9

  if (isNaN(count) || count < 1) {
    showNotification("生成数量至少为1", "error")
    return
  }

  if (!trainingCorpus.trim()) {
    showNotification("请输入训练语料", "error")
    return
  }

  // Show loading state
  showLoading()

  // 准备API请求数据
  const requestData = {
    input: trainingCorpus,
    prompt_temp: selfEvalPrompt,
    generate_num: count,
    temperature: temperature,
    top_p: topP,
    top_k: topK,
  }

  console.log("发送样本生成请求数据:", requestData)

  // 调用样本生成API
      fetch(API_CONFIG.getUrl('GENERATE_SAMPLES'), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestData),
    mode: "cors",
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("API响应错误: " + response.status)
      }
      return response.json()
    })
    .then((data) => {
      if (data.code === 200) {
        // 处理API返回的样本数据
        processSamplesResponse(data.data)
      } else {
        showNotification("生成样本失败: " + data.message, "error")
        hideLoading()
      }
    })
    .catch((error) => {
      console.error("请求样本生成失败:", error)
      alert("请求样本生成失败: " + error.message)
      hideLoading()
    })
}

// 处理样本生成API响应
function processSamplesResponse(apiData) {
  console.log("API返回的样本数据:", apiData)

  // 构建样本数据结构 - 只显示普通样本
  const samples = apiData.samples.map((sample, index) => ({
    id: index + 1,
    content: sample,
    type: "text",
    quality: "standard",
  }))

  // 显示结果
  displayResults(samples, apiData.advantage_sample)
}

// Show loading state
function showLoading() {
  document.getElementById("resultsContent").innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <span>正在生成样本，请稍候...</span>
        </div>
    `

  document.getElementById("advantageSamplesContent").innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <span>统计中...</span>
        </div>
    `
}

function hideLoading() {
  document.getElementById("resultsContent").innerHTML = `
        <div class="empty-state">
            <i class="fas fa-exclamation-triangle"></i>
            <p>生成失败，请重试</p>
        </div>
    `

  document.getElementById("advantageSamplesContent").innerHTML = `
        <div class="empty-state">
            <i class="fas fa-exclamation-triangle"></i>
            <p>统计失败</p>
        </div>
    `
}

// Display results
function displayResults(samples, advantageSample = null) {
  generatedSamples = samples
  currentAdvantageSample = advantageSample

  const resultsContent = document.getElementById("resultsContent")
  const advantageSamplesContent = document.getElementById("advantageSamplesContent")

  // Display regular samples with improved layout
  resultsContent.innerHTML = samples
    .map((sample) => {
      return `
            <div class="sample-item">
                <div class="sample-header">
                    <div class="sample-id">样本 ${sample.id}</div>
                    <div class="sample-actions">
                        <button class="copy-btn" onclick="copySampleWithFeedback('${sample.id}', this)">
                            <i class="fas fa-copy"></i>
                        </button>
                    </div>
                </div>
                <div class="sample-content">${sample.content}</div>
            </div>
        `
    })
    .join("")

  // Display advantage sample separately
  if (advantageSample) {
    advantageSamplesContent.innerHTML = `
            <div class="advantage-sample-item">
                <div class="advantage-sample-header">
                    <div class="advantage-sample-title">
                        <i class="fas fa-star"></i>
                        优势样本
                    </div>
                    <div class="sample-actions">
                        <button class="copy-btn" onclick="copyAdvantageWithFeedback(this)">
                            <i class="fas fa-copy"></i>
                        </button>
                    </div>
                </div>
                <div class="advantage-sample-content">${advantageSample}</div>
            </div>
        `
  } else {
    advantageSamplesContent.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-star"></i>
                <p>暂无优势样本</p>
            </div>
        `
  }
}

// Copy all results
function copyAllResults(buttonElement) {
  if (generatedSamples.length === 0) {
    showNotification("没有可复制的结果", "error")
    return
  }

  // 只复制样本输出的内容，不包含优势样本
  const allText = generatedSamples
    .map((sample) => {
      let content = ""
      if (sample.type === "conversation") {
        content = sample.content.map((turn) => `${turn.speaker}: ${turn.message}`).join("\n")
      } else if (sample.type === "data") {
        content = JSON.stringify(sample.content, null, 2)
      } else {
        content = sample.content
      }
      return `=== 样本 ${sample.id} ===\n${content}`
    })
    .join("\n\n")

  copyTextWithFeedback(allText, buttonElement)
}

// Export results
function exportResults() {
  if (generatedSamples.length === 0) {
    showNotification("没有可导出的结果", "error")
    return
  }

  // Default to text format since we've removed the output format selector
  const format = "text"
  let content = ""
  let filename = ""
  let mimeType = ""

  switch (format) {
    case "json":
      content = JSON.stringify(generatedSamples, null, 2)
      filename = `samples_${new Date().toISOString().split("T")[0]}.json`
      mimeType = "application/json"
      break
    case "csv":
      content = generateCSV(generatedSamples)
      filename = `samples_${new Date().toISOString().split("T")[0]}.csv`
      mimeType = "text/csv"
      break
    default:
      content = generatedSamples
        .map((sample) => {
          let sampleContent = ""
          if (sample.type === "conversation") {
            sampleContent = sample.content.map((turn) => `${turn.speaker}: ${turn.message}`).join("\n")
          } else if (sample.type === "data") {
            sampleContent = JSON.stringify(sample.content, null, 2)
          } else {
            sampleContent = sample.content
          }
          return `=== ${sample.id} ===\n${sampleContent}`
        })
        .join("\n\n")
      filename = `samples_${new Date().toISOString().split("T")[0]}.txt`
      mimeType = "text/plain"
  }

  const blob = new Blob([content], { type: `${mimeType};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// Generate CSV
function generateCSV(samples) {
  const headers = ["ID", "Type", "Content", "Metadata"]
  const rows = samples.map((sample) => [
    sample.id,
    sample.type,
    JSON.stringify(sample.content).replace(/"/g, '""'),
    JSON.stringify(sample.metadata).replace(/"/g, '""'),
  ])

  return [headers, ...rows].map((row) => row.map((cell) => `"${cell}"`).join(",")).join("\n")
}

// Clear results
function clearResults() {
  generatedSamples = []

  document.getElementById("resultsContent").innerHTML = `
        <div class="empty-state">
            <i class="fas fa-magic"></i>
            <p>点击"生成样本"按钮开始生成</p>
        </div>
    `

  document.getElementById("advantageSamplesContent").innerHTML = `
        <div class="empty-state">
            <i class="fas fa-trophy"></i>
            <p>优势样本将在生成后显示</p>
        </div>
    `
}

// 页面加载完成时初始化
document.addEventListener("DOMContentLoaded", () => {
  // 初始化模型选择器
  initModelDropdown()

  console.log("样本生成页面已加载")

  // Set currentSampleType to text as default
  currentSampleType = "text"
})

// Reset form function
function resetForm() {
  // Reset sample count
  document.getElementById("sampleCount").value = 5

  // Reset parameters
  document.getElementById("temperature").value = 0.1
  document.getElementById("topK").value = 0
  document.getElementById("topP").value = 1

  // Reset training corpus but preserve self-evaluation prompt
  document.getElementById("trainingCorpus").value = ""

  // Clear results
  clearResults()
}

// Generate with dynamic entropy temperature
function generateWithDynamicEntropy() {
  const count = Number.parseInt(document.getElementById("sampleCount").value)
  const trainingCorpus = document.getElementById("trainingCorpus").value
  const selfEvalPrompt = document.getElementById("selfEvaluationPrompt").value
  const temperature = Number.parseFloat(document.getElementById("temperature").value) || 0.1
  const topK = Number.parseInt(document.getElementById("topK").value) || 0
  const topP = Number.parseFloat(document.getElementById("topP").value) || 1

  if (isNaN(count) || count < 1) {
    showNotification("生成数量至少为1", "error")
    return
  }

  if (!trainingCorpus.trim()) {
    showNotification("请输入训练语料", "error")
    return
  }

  // Show loading state
  showLoading()

  // 准备动态熵生成API请求数据
  const requestData = {
    input: trainingCorpus,
    prompt_temp: selfEvalPrompt,
    generate_num: count,
    temperature: temperature,
    top_p: topP,
    top_k: topK,
  }

  console.log("发送动态熵生成请求数据:", requestData)

  // 调用动态熵生成API
      fetch(API_CONFIG.getUrl('DYNAMIC_GENERATE_SAMPLES'), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestData),
    mode: "cors",
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("API响应错误: " + response.status)
      }
      return response.json()
    })
    .then((data) => {
      if (data.code === 200) {
        // 处理API返回的动态熵样本数据
        processSamplesResponse(data.data)
      } else {
        showNotification("动态熵生成失败: " + data.message, "error")
        hideLoading()
      }
    })
    .catch((error) => {
      console.error("请求动态熵生成失败:", error)
      showNotification("请求动态熵生成失败: " + error.message, "error")
      hideLoading()
    })
}

// Load training corpus sample
function loadTrainingCorpusSample() {
  const sampleText = `我们需要设计一个更好的用户界面。界面应该简洁、直观且易于使用。
用户界面的颜色应该协调，不要太花哨。
所有按钮和链接应该清晰可见，不要隐藏在页面角落。
导航菜单应该逻辑清晰，用户能够轻松找到所需功能。
响应式设计很重要，确保在移动设备上也能良好显示。
用户反馈是改进界面的重要依据，我们应该收集并分析用户行为数据。
简化流程，减少用户完成任务所需的点击次数。
提供清晰的错误提示，帮助用户理解并解决问题。
界面加载速度要快，不要让用户等待太久。
最后，确保界面符合品牌形象和设计规范。`
  document.getElementById("trainingCorpus").value = sampleText
}

// 添加proxyFetch函数（用于模型选择器API调用）
function proxyFetch(url) {
  console.log("发起请求到:", url)

  return fetch(url, {
    mode: "cors",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  })
    .then((response) => {
      console.log("收到响应状态码:", response.status)
      if (!response.ok) {
        throw new Error(`HTTP错误: ${response.status}`)
      }
      return response.json()
    })
    .then((data) => {
      console.log("响应数据:", data)
      if (data && data.code !== 200) {
        console.warn("API响应码非200:", data.code, data.message)
      }
      return data
    })
    .catch((error) => {
      console.error("请求失败详情:", error.message)
      return {
        code: 500,
        message: "请求失败: " + error.message,
        error: true,
      }
    })
}

// 模型选择器初始化
function initModelDropdown() {
  const modelDropdown = document.getElementById("modelDropdown")
  const modelSelect = document.getElementById("modelSelect")
  const modelCurrent = modelDropdown.querySelector(".model-current")
  const modelNameDisplay = modelDropdown.querySelector(".model-name")

  // 清空现有选项（确保即使在请求失败时也不会有默认模型）
  modelSelect.innerHTML = ""
  const modelMenu = modelDropdown.querySelector(".model-menu")
  modelMenu.innerHTML = ""

  // 显示加载状态
  modelNameDisplay.textContent = "加载中..."

  // 获取当前模型名称
          proxyFetch(API_CONFIG.getUrl('CURRENT_MODEL_NAME'))
    .then((data) => {
      console.log("获取当前模型返回数据:", data)
      if (data.code === 200) {
        // 显示当前模型名称
        const currentModelName = data.data.model_name
        modelNameDisplay.textContent = getShortModelName(currentModelName)
        console.log("当前模型:", currentModelName)

        // 获取所有模型列表
        getModelList(currentModelName)
      } else {
        console.error("获取当前模型失败:", data.message)
        modelNameDisplay.textContent = "获取模型失败"
        // 仍然尝试获取模型列表
        getModelList()
      }
    })
    .catch((error) => {
      console.error("获取当前模型请求失败:", error)
      modelNameDisplay.textContent = "请求错误"

      // 仍然尝试获取所有模型列表
      getModelList()
    })

  // 获取所有模型列表的函数
  function getModelList(currentModelName) {
    proxyFetch(API_CONFIG.getUrl('GET_MODELS'))
      .then((data) => {
        console.log("获取所有模型返回数据:", data)
        if (data.code === 200) {
          const models = data.data.model_name

          // 清空现有选项
          modelSelect.innerHTML = ""
          modelMenu.innerHTML = ""

          if (models && models.length > 0) {
            // 添加所有模型选项
            models.forEach((model, index) => {
              // 添加到隐藏的select元素
              const option = document.createElement("option")
              option.value = model
              option.textContent = getShortModelName(model)
              if (currentModelName && model === currentModelName) {
                option.selected = true
              }
              modelSelect.appendChild(option)

              // 添加到自定义下拉菜单
              const menuItem = document.createElement("div")
              menuItem.className = "model-item"
              if (currentModelName && model === currentModelName) {
                menuItem.classList.add("active")
              }
              menuItem.setAttribute("data-value", model)
              menuItem.innerHTML = `<span>${getShortModelName(model)}</span>`

              menuItem.addEventListener("click", function (e) {
                e.stopPropagation()

                // 切换模型
                changeModel(model, this)
              })

              modelMenu.appendChild(menuItem)
            })
            console.log("成功加载模型列表:", models.length, "个模型")
          } else {
            console.warn("获取的模型列表为空")
            modelNameDisplay.textContent = "无可用模型"
          }
        } else {
          console.error("获取模型列表失败:", data.message)
          modelNameDisplay.textContent = "获取列表失败"
        }
      })
      .catch((error) => {
        console.error("获取模型列表请求失败:", error)
        modelNameDisplay.textContent = "请求错误"
        // 不添加默认模型
      })
  }

  // 点击当前选中的模型，显示/隐藏下拉菜单
  modelCurrent.addEventListener("click", (e) => {
    e.stopPropagation()
    // 只有当有模型选项时才切换下拉菜单
    if (modelMenu.children.length > 0) {
      modelDropdown.classList.toggle("active")
    } else {
      // 如果没有模型选项，显示提示
      console.log("没有可用的模型选项")
    }
  })

  // 点击页面其他地方关闭下拉菜单
  document.addEventListener("click", () => {
    modelDropdown.classList.remove("active")
  })
}

// 切换模型
function changeModel(modelName, menuItem) {
  // 显示加载状态
  const modelNameDisplay = document.querySelector(".model-name")
  const originalText = modelNameDisplay.textContent
  modelNameDisplay.textContent = "切换中..."

  console.log("尝试切换到模型:", modelName)

  // 构造URL，确保modelName被正确编码
      const url = new URL(API_CONFIG.getUrl('CHANGE_MODEL'))
  // url.searchParams.append("model_name", modelName)

  console.log("完整请求URL:", url.toString())
  $ajaxGet(url,{model_name:modelName},function(data){
    console.log("切换模型返回数据:", data)
      if (data.code === 200) {
        // 更新当前显示
        modelNameDisplay.textContent = getShortModelName(modelName)

        // 更新隐藏的select元素值
        const modelSelect = document.getElementById("modelSelect")
        modelSelect.value = modelName

        // 更新激活状态
        const modelMenu = document.querySelector(".model-menu")
        const modelItems = modelMenu.querySelectorAll(".model-item")
        modelItems.forEach((mi) => mi.classList.remove("active"))
        menuItem.classList.add("active")

        // 关闭下拉菜单
        document.getElementById("modelDropdown").classList.remove("active")

        console.log("模型成功切换为:", modelName)
      } else {
        console.error("模型切换失败:", data.message)
        modelNameDisplay.textContent = originalText
        alert("模型切换失败: " + data.message)
      }
  },function(err){
    modelNameDisplay.textContent = originalText
  })


}

// 获取模型的短名称（从路径中提取）
function getShortModelName(modelPath) {
  if (!modelPath) return "Unknown"

  // 移除路径末尾的斜杠
  const cleanPath = modelPath.endsWith("/") ? modelPath.slice(0, -1) : modelPath

  // 提取最后一个斜杠后的部分作为模型名
  const parts = cleanPath.split("/")
  return parts[parts.length - 1]
}

// 复制功能 - 从attention页面复制
function copySampleWithFeedback(sampleId, buttonElement) {
  // 确保sampleId是数字类型进行比较
  const numericSampleId = Number.parseInt(sampleId)
  const sample = generatedSamples.find((s) => s.id === numericSampleId)
  if (sample) {
    copyTextWithFeedback(sample.content, buttonElement)
  } else {
    console.error(
      "找不到样本:",
      sampleId,
      "可用样本:",
      generatedSamples.map((s) => s.id),
    )
  }
}

function copyAdvantageWithFeedback(buttonElement) {
  if (currentAdvantageSample) {
    copyTextWithFeedback(currentAdvantageSample, buttonElement)
  }
}

function copyTextWithFeedback(text, buttonElement) {
  if (navigator.clipboard) {
    navigator.clipboard
      .writeText(text)
      .then(() => showCopySuccess(buttonElement))
      .catch((err) => {
        console.error("复制失败:", err)
        // 如果现代API失败，回退到兼容模式
        fallbackCopy(text, buttonElement)
      })
  } else {
    // 回退到兼容模式
    fallbackCopy(text, buttonElement)
  }
}

// 回退到兼容模式的复制方法
function fallbackCopy(text, buttonElement) {
  // 创建一个临时textarea元素来复制文本
  const textarea = document.createElement("textarea")
  textarea.value = text
  textarea.style.position = "fixed" // 确保在屏幕外
  document.body.appendChild(textarea)
  textarea.select()

  try {
    // 执行复制命令
    const successful = document.execCommand("copy")
    if (successful) {
      showCopySuccess(buttonElement)
    }
  } catch (err) {
    console.error("复制失败:", err)
  }

  // 移除临时元素
  document.body.removeChild(textarea)
}

// 显示复制成功的视觉反馈
function showCopySuccess(buttonElement) {
  if (buttonElement) {
    // 只对特定按钮应用动画
    const originalIcon = buttonElement.innerHTML
    buttonElement.innerHTML = '<i class="fas fa-check"></i>'
    buttonElement.classList.add("success")

    // 500毫秒后恢复原始状态（更快的恢复）
    setTimeout(() => {
      buttonElement.innerHTML = originalIcon
      buttonElement.classList.remove("success")
    }, 500)
  }

  console.log("复制成功")
}

// Show notification function
function showNotification(message, type = "info") {
  // Create notification element
  const notification = document.createElement("div")
  notification.className = "notification"
  notification.style.position = "fixed"
  notification.style.top = "20px"
  notification.style.right = "20px"
  notification.style.padding = "12px 20px"
  notification.style.borderRadius = "4px"
  notification.style.fontSize = "14px"
  notification.style.fontWeight = "500"
  notification.style.zIndex = "2000"
  notification.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.15)"
  notification.style.transition = "opacity 0.3s ease-in-out, transform 0.3s ease-in-out"
  notification.style.opacity = "0"
  notification.style.transform = "translateY(-10px)"

  // Set style based on type
  if (type === "success") {
    notification.style.backgroundColor = "#10b981"
    notification.style.color = "white"
    notification.innerHTML = `<i class="fas fa-check-circle" style="margin-right: 8px;"></i> ${message}`
  } else if (type === "error") {
    notification.style.backgroundColor = "#ef4444"
    notification.style.color = "white"
    notification.innerHTML = `<i class="fas fa-exclamation-circle" style="margin-right: 8px;"></i> ${message}`
  } else {
    notification.style.backgroundColor = "#3b82f6"
    notification.style.color = "white"
    notification.innerHTML = `<i class="fas fa-info-circle" style="margin-right: 8px;"></i> ${message}`
  }

  // Add to DOM
  document.body.appendChild(notification)

  // Animate in
  setTimeout(() => {
    notification.style.opacity = "1"
    notification.style.transform = "translateY(0)"
  }, 10)

  // Remove after timeout
  setTimeout(() => {
    notification.style.opacity = "0"
    notification.style.transform = "translateY(-10px)"

    setTimeout(() => {
      document.body.removeChild(notification)
    }, 300) // Wait for fade out animation
  }, 3000) // Show for 3 seconds
}
