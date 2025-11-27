let currentSubModule = "attention-score"
let currentResultTab = "original"
let currentAnalysisAnswer = "model" // 'model' or 'target'
let analysisData = null
let selectedToken = null

// Initialize analysis area visibility on page load
document.addEventListener("DOMContentLoaded", () => {
  // Wait a bit to ensure all elements are rendered
  setTimeout(() => {
    updateAnalysisAreaVisibility(currentSubModule)

    // 初始化模型选择器
    initModelDropdown()
  }, 100)

  // Add keyboard support for removing score displays
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      const containers = document.querySelectorAll(".token-score-display-container")
      containers.forEach((el) => {
        if (el.parentNode) el.parentNode.removeChild(el)
      })
    }
  })
})

// 添加proxyFetch函数
function proxyFetch(url) {
  console.log("发起请求到:", url)

  // 不使用corsproxy.io，尝试直接请求
  return fetch(url, {
    // 添加模式以便浏览器尝试跳过CORS检查
    mode: "cors",
    // 添加更多的header可能帮助绕过某些CORS限制
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  })
    .then((response) => {
      console.log("收到响应状态码:", response.status)
      // 检查HTTP状态
      if (!response.ok) {
        throw new Error(`HTTP错误: ${response.status}`)
      }
      // 尝试解析为JSON并返回promise
      return response.json()
    })
    .then((data) => {
      // 检查API响应的code字段
      console.log("响应数据:", data)
      if (data && data.code !== 200) {
        console.warn("API响应码非200:", data.code, data.message)
      }
      return data
    })
    .catch((error) => {
      // 更详细的错误记录
      console.error("请求失败详情:", error.message)
      // 返回一个带有详细错误信息的对象
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

  try {
    // 获取基础URL并验证
    const baseUrl = API_CONFIG.getUrl('CHANGE_MODEL');
    console.log("基础URL:", baseUrl);
    console.log("当前环境:", {
      hostname: window.location.hostname,
      protocol: window.location.protocol,
      host: window.location.host
    });
    
    // 构造URL，确保modelName被正确编码
    const url = new URL(baseUrl);
    // url.searchParams.append("model_name", modelName);

    console.log("完整请求URL:", url.toString());
    $ajaxGet(url,{model_name:modelName},function(data){
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

    // 使用新的请求URL
    // proxyFetch(url.toString())
    //   .then((data) => {
    //     console.log("切换模型返回数据:", data)
    //     if (data.code === 200) {
    //       // 更新当前显示
    //       modelNameDisplay.textContent = getShortModelName(modelName)

    //       // 更新隐藏的select元素值
    //       const modelSelect = document.getElementById("modelSelect")
    //       modelSelect.value = modelName

    //       // 更新激活状态
    //       const modelMenu = document.querySelector(".model-menu")
    //       const modelItems = modelMenu.querySelectorAll(".model-item")
    //       modelItems.forEach((mi) => mi.classList.remove("active"))
    //       menuItem.classList.add("active")

    //       // 关闭下拉菜单
    //       document.getElementById("modelDropdown").classList.remove("active")

    //       console.log("模型成功切换为:", modelName)
    //     } else {
    //       console.error("模型切换失败:", data.message)
    //       modelNameDisplay.textContent = originalText
    //       alert("模型切换失败: " + data.message)
    //     }
    //   })
    //   .catch((error) => {
    //     console.error("模型切换请求失败:", error)
    //     modelNameDisplay.textContent = originalText
    //     alert("模型切换请求失败，可能是网络问题或API服务不可用")
    //   })
  } catch (error) {
    console.error("URL构造失败:", error);
    console.error("基础URL:", API_CONFIG.getUrl('CHANGE_MODEL'));
    
    // 恢复原始文本并显示错误
    modelNameDisplay.textContent = originalText;
    // alert("切换模型失败：URL配置错误。请检查网络连接或联系管理员。");
    return;
  }
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

// Sub module switching
function switchSubModule(subModule) {
  currentSubModule = subModule

  // Update active sub module button
  document.querySelectorAll(".sub-module-btn").forEach((btn) => {
    btn.classList.remove("active")
  })
  
  // 找到正确的按钮元素（处理点击图标或文字的情况）
  const clickedElement = event.target
  const buttonElement = clickedElement.closest('.sub-module-btn')
  if (buttonElement) {
    buttonElement.classList.add("active")
  }

  // Update placeholder text based on module
  const inputContent = document.getElementById("inputContent")
  const targetContent = document.getElementById("targetContent")

  if (subModule === "token-loss") {
    inputContent.placeholder = "请输入问题或提示内容，系统将分析每个token的损失值..."
    targetContent.placeholder = "请输入您期望的目标答案..."
  } else if (subModule === "token-entropy") {
    inputContent.placeholder = "请输入问题或提示内容，系统将分析每个token的熵值..."
    targetContent.placeholder = "请输入您期望的目标答案..."
  } else {
    inputContent.placeholder = "请输入问题或提示内容，模型将基于此生成答案..."
    targetContent.placeholder = "请输入您期望的目标答案..."
  }

  // Reset analysis when switching modules (always clear content)
  resetAnalysis()

  // Update dynamic analysis area visibility
  updateAnalysisAreaVisibility(subModule)
}

// Update analysis area visibility based on current sub-module
function updateAnalysisAreaVisibility(subModule) {
  // Get all analysis areas
  const attentionLayout = document.getElementById("attentionAnalysisLayout")
  const dynamicAnalysisArea = document.getElementById("dynamicAnalysisArea")
  const resultsDisplay = document.getElementById("resultsDisplay")

  // Check if elements exist
  if (!attentionLayout || !dynamicAnalysisArea || !resultsDisplay) {
    console.error("Analysis area elements not found")
    return
  }

  // Update title and description based on sub-module
  const titleElement = document.getElementById("resultsTitle")
  const descriptionElement = document.getElementById("resultsDescription")
  const modelChartTitle = document.getElementById("modelChartTitle")
  const targetChartTitle = document.getElementById("targetChartTitle")

  // 移除所有分析类型的CSS类
  resultsDisplay.classList.remove("attention-score", "token-loss", "token-entropy")

  if (subModule === "token-loss") {
    // 切换到损失分析布局
    attentionLayout.style.display = "none"
    dynamicAnalysisArea.style.display = "block"

    // 添加特定分析类型的CSS类
    resultsDisplay.classList.add("token-loss")

    // 更新柱状图标题
    if (modelChartTitle) modelChartTitle.textContent = "模型生成答案"
    if (targetChartTitle) targetChartTitle.textContent = "目标期望答案"

    titleElement.textContent = "Token损失分析对比"
    titleElement.style.color = "#dc2626"
    descriptionElement.innerHTML = `
            <i class="fas fa-exclamation-triangle" style="color: #dc2626;"></i>
            对比模型答案和目标答案的Token损失分布，损失值越高表示预测难度越大
        `
  } else if (subModule === "token-entropy") {
    // 切换到熵分析布局
    attentionLayout.style.display = "none"
    dynamicAnalysisArea.style.display = "block"

    // 添加特定分析类型的CSS类
    resultsDisplay.classList.add("token-entropy")

    // 更新柱状图标题
    if (modelChartTitle) modelChartTitle.textContent = "模型生成答案"
    if (targetChartTitle) targetChartTitle.textContent = "目标期望答案"

    titleElement.textContent = "Token熵分析对比"
    titleElement.style.color = "#2563eb"
    descriptionElement.innerHTML = `
            <i class="fas fa-chart-area" style="color: #2563eb;"></i>
            对比模型答案和目标答案的Token熵分布，熵值越高表示选择不确定性越大
        `
  } else {
    // 切换到注意力分析布局
    attentionLayout.style.display = "grid"
    dynamicAnalysisArea.style.display = "none"

    // 添加特定分析类型的CSS类
    resultsDisplay.classList.add("attention-score")

    titleElement.textContent = "注意力分数对比分析"
    titleElement.style.color = "#374151"
    descriptionElement.innerHTML = `
            <i class="fas fa-info-circle"></i>
            <strong>使用步骤：</strong>1️⃣ 先点击左右两侧答案中的token（或All按钮）查看注意力高亮；2️⃣ 再点击对应下方高亮的问题token查看详细分数。<br>
            <small style="color: #6b7280; margin-top: 4px; display: block;"><i class="fas fa-lightbulb"></i> 提示：左右两侧独立操作，互不影响。分数显示在对应区域下方。</small>
        `
  }
}

// 此函数在新布局中不再使用，但保留以便向后兼容
function selectAnswer(answerType) {
  currentAnalysisAnswer = answerType

  // 只有在Token损失和Token熵分析中使用这个功能
  if (currentSubModule !== "attention-score") {
    // Update button states
    document.querySelectorAll(".selector-btn").forEach((btn) => {
      btn.classList.remove("active")
    })

    const targetBtn = document.querySelector(`[data-answer="${answerType}"]`)
    if (targetBtn) {
      targetBtn.classList.add("active")
    }

    // Update analysis type indicator
    const indicator = document.getElementById("currentAnalysisType")
    if (indicator) {
      if (answerType === "model") {
        indicator.textContent = "(分析模型生成答案)"
      } else {
        indicator.textContent = "(分析目标期望答案)"
      }
    }

    // Update analysis info if data is available
    if (analysisData) {
      updateAnalysisInfo(analysisData)
    }
  }
}

// Result tab switching
function switchResultTab(tab) {
  currentResultTab = tab

  // Update active tab button
  document.querySelectorAll(".tab-button").forEach((btn) => {
    btn.classList.remove("active")
  })
  event.target.classList.add("active")

  // Update title
  const title = tab === "original" ? "模型生成答案" : "目标期望答案"
  document.getElementById("currentOutputTitle").textContent = title

  // Re-render if we have data
  if (analysisData) {
    displayAnalysisResults(analysisData)
  }
}

// Generate analysis
function generateAnalysis() {
  const inputContent = document.getElementById("inputContent").value.trim()
  const targetContent = document.getElementById("targetContent").value.trim()

  if (!inputContent) {
    showNotification("请先输入问题或提示内容", "error")
    return
  }

  if (!targetContent) {
    showNotification("请先输入目标期望答案", "error")
    return
  }

  // Show loading state
  showLoading()

  if (currentSubModule === "attention-score") {
    // 使用真实API请求注意力分数
    const requestData = {
      input: inputContent,
      output: targetContent,
    }

    console.log("发送的请求数据:", requestData)

    // 调用API获取注意力分数
            fetch(API_CONFIG.getUrl('ATTENTION_SCORES'), {
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
          // 处理API返回的注意力分数数据
          processAttentionScoresResponse(data.data, inputContent, targetContent)
        } else {
          showNotification("获取注意力分数失败: " + data.message, "error")
          resetLoadingState()
        }
      })
      .catch((error) => {
        console.error("请求注意力分数失败:", error)
        showNotification("请求注意力分数失败: " + error.message, "error")
        resetLoadingState()
      })
  } else if (currentSubModule === "token-loss") {
    // 使用真实API请求token损失分数
    const requestData = {
      input: inputContent,
      output: targetContent,
    }

    console.log("发送token损失分析请求数据:", requestData)

    // 调用API获取token损失分数
            fetch(API_CONFIG.getUrl('LOSS_SCORES'), {
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
          // 处理API返回的token损失数据
          processLossScoresResponse(data.data, inputContent, targetContent)
        } else {
          showNotification("获取token损失分数失败: " + data.message, "error")
          resetLoadingState()
        }
      })
      .catch((error) => {
        console.error("请求token损失分数失败:", error)
        showNotification("请求token损失分数失败: " + error.message, "error")
        resetLoadingState()
      })
  } else if (currentSubModule === "token-entropy") {
    // 使用真实API请求token熵分数
    const requestData = {
      input: inputContent,
      output: targetContent,
    }

    console.log("发送token熵分析请求数据:", requestData)

    // 调用API获取token熵分数
            fetch(API_CONFIG.getUrl('ENTROPY_SCORES'), {
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
          // 处理API返回的token熵数据
          processEntropyScoresResponse(data.data, inputContent, targetContent)
        } else {
          showNotification("获取token熵分数失败: " + data.message, "error")
          resetLoadingState()
        }
      })
      .catch((error) => {
        console.error("请求token熵分数失败:", error)
        showNotification("请求token熵分数失败: " + error.message, "error")
        resetLoadingState()
      })
  }
}

// 处理注意力分数API响应
function processAttentionScoresResponse(apiData, inputContent, targetContent) {
  console.log("API返回的数据:", apiData)
  console.log("输入tokens数量:", apiData.input_tokens.length)
  console.log("模型输出tokens数量:", apiData.origin_output_tokens.length)
  console.log("目标输出tokens数量:", apiData.output_tokens ? apiData.output_tokens.length : 0)
  console.log("模型注意力分数数量:", apiData.origin_attention_scores.length)
  console.log("目标注意力分数数量:", apiData.attention_scores ? apiData.attention_scores.length : 0)

  // 构建分析数据结构
  const analysisData = {
    input: apiData.input_tokens,
    inputText: inputContent,
    modelAnswer: apiData.origin_output_tokens.join(""),
    targetAnswer: apiData.output_tokens.join(""),
    original: [],
    target: [],
    model: document.getElementById("modelSelect").value,
    analysisType: "attention-score",
  }

  // 处理模型原始输出token及其注意力分数
  analysisData.original = apiData.origin_output_tokens.map((token, index) => {
    // 获取该token对应的注意力分数数组
    const attentionScores = apiData.origin_attention_scores[index] || []
    const avgAttentionScore =
      attentionScores.length > 0 ? attentionScores.reduce((sum, score) => sum + score, 0) / attentionScores.length : 0

    return {
      id: index,
      text: token,
      attentionScore: avgAttentionScore,
      position: index,
      inputAttention: attentionScores.length > 0 ? attentionScores : apiData.input_tokens.map(() => 0),
    }
  })

  // 处理目标输出token及其注意力分数
  if (apiData.output_tokens && apiData.attention_scores) {
    analysisData.target = apiData.output_tokens.map((token, index) => {
      // 获取该token对应的注意力分数数组
      const attentionScores = apiData.attention_scores[index] || []
      const avgAttentionScore =
        attentionScores.length > 0 ? attentionScores.reduce((sum, score) => sum + score, 0) / attentionScores.length : 0

      return {
        id: index,
        text: token,
        attentionScore: avgAttentionScore,
        position: index,
        inputAttention: attentionScores.length > 0 ? attentionScores : apiData.input_tokens.map(() => 0),
      }
    })
  } else {
    // 如果API没有返回目标token数据，创建空数组
    console.warn("API未返回output_tokens或attention_scores")
    analysisData.target = []
  }

  // 显示分析结果
  displayAnalysisResults(analysisData)
}

// 处理token损失分数API响应
function processLossScoresResponse(apiData, inputContent, targetContent) {
  console.log("API返回的损失数据:", apiData)
  console.log("模型输出tokens数量:", apiData.origin_output_tokens.length)
  console.log("目标输出tokens数量:", apiData.output_tokens ? apiData.output_tokens.length : 0)
  console.log("模型损失分数数量:", apiData.origin_loss_per_token.length)
  console.log("目标损失分数数量:", apiData.loss_per_token ? apiData.loss_per_token.length : 0)

  // 构建分析数据结构
  const analysisData = {
    input: [inputContent], // 损失分析不需要input tokens，但保持结构一致
    inputText: inputContent,
    modelAnswer: apiData.origin_output_tokens.join(""),
    targetAnswer: apiData.output_tokens.join(""),
    original: [],
    target: [],
    model: document.getElementById("modelSelect").value,
    analysisType: "token-loss",
  }

  // 处理模型原始输出token及其损失分数
  analysisData.original = apiData.origin_output_tokens.map((token, index) => {
    const lossValue = apiData.origin_loss_per_token[index] || 0

    return {
      id: index,
      text: token,
      lossValue: lossValue,
      position: index,
    }
  })

  // 处理目标输出token及其损失分数
  if (apiData.output_tokens && apiData.loss_per_token) {
    analysisData.target = apiData.output_tokens.map((token, index) => {
      const lossValue = apiData.loss_per_token[index] || 0

      return {
        id: index,
        text: token,
        lossValue: lossValue,
        position: index,
      }
    })
  } else {
    // 如果API没有返回目标token数据，创建空数组
    console.warn("API未返回output_tokens或loss_per_token")
    analysisData.target = []
  }

  // 显示分析结果
  displayAnalysisResults(analysisData)
}

// 处理token熵分数API响应
function processEntropyScoresResponse(apiData, inputContent, targetContent) {
  console.log("API返回的熵数据:", apiData)
  console.log("模型输出tokens数量:", apiData.origin_output_tokens.length)
  console.log("目标输出tokens数量:", apiData.output_tokens ? apiData.output_tokens.length : 0)
  console.log("模型熵分数数量:", apiData.origin_loss_per_token.length)
  console.log("目标熵分数数量:", apiData.loss_per_token ? apiData.loss_per_token.length : 0)

  // 构建分析数据结构
  const analysisData = {
    input: [inputContent], // 熵分析不需要input tokens，但保持结构一致
    inputText: inputContent,
    modelAnswer: apiData.origin_output_tokens.join(""),
    targetAnswer: apiData.output_tokens.join(""),
    original: [],
    target: [],
    model: document.getElementById("modelSelect").value,
    analysisType: "token-entropy",
  }

  // 处理模型原始输出token及其熵分数
  analysisData.original = apiData.origin_output_tokens.map((token, index) => {
    const entropyValue = apiData.origin_loss_per_token[index] || 0

    return {
      id: index,
      text: token,
      entropyValue: entropyValue,
      position: index,
    }
  })

  // 处理目标输出token及其熵分数
  if (apiData.output_tokens && apiData.loss_per_token) {
    analysisData.target = apiData.output_tokens.map((token, index) => {
      const entropyValue = apiData.loss_per_token[index] || 0

      return {
        id: index,
        text: token,
        entropyValue: entropyValue,
        position: index,
      }
    })
  } else {
    // 如果API没有返回目标token数据，创建空数组
    console.warn("API未返回output_tokens或loss_per_token")
    analysisData.target = []
  }

  // 显示分析结果
  displayAnalysisResults(analysisData)
}

// 重置加载状态（当API请求失败时）
function resetLoadingState() {
  if (currentSubModule === "attention-score") {
    document.getElementById("modelTokens").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>请求数据失败，请重试</p>
            </div>
        `
    document.getElementById("targetTokens").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>请求数据失败，请重试</p>
            </div>
        `
    document.getElementById("modelInputMirror").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>请求数据失败，请重试</p>
            </div>
        `
    document.getElementById("targetInputMirror").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>请求数据失败，请重试</p>
            </div>
        `
  } else if (currentSubModule === "token-loss" || currentSubModule === "token-entropy") {
    document.getElementById("modelChart").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>请求数据失败，请重试</p>
            </div>
        `
    document.getElementById("targetChart").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>请求数据失败，请重试</p>
            </div>
        `
  }
}

// Reset analysis
function resetAnalysis() {
  // 不清空输入框内容，只清理分析结果
  // document.getElementById("inputContent").value = ""
  // document.getElementById("targetContent").value = ""

  // 根据当前分析类型重置对应区域
  if (currentSubModule === "attention-score") {
    // 重置模型答案区域
    document.getElementById("modelTokens").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-robot"></i>
                <p>模型生成的答案将在这里显示</p>
            </div>
        `

    // 重置目标答案区域
    document.getElementById("targetTokens").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-bullseye"></i>
                <p>目标期望答案将在这里显示</p>
            </div>
        `

    // 重置所有注意力高亮区域
    document.getElementById("modelInputMirror").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-question-circle"></i>
                <p>点击上方模型答案的token查看注意力分数</p>
            </div>
        `

    document.getElementById("targetInputMirror").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-question-circle"></i>
                <p>点击上方目标答案的token查看注意力分数</p>
            </div>
        `
  } else {
    // 重置dynamicAnalysisArea的chart容器（损失和熵分析）
    document.getElementById("modelChart").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-chart-bar"></i>
                <p>模型答案柱状图将在这里显示</p>
            </div>
        `

    document.getElementById("targetChart").innerHTML = `
            <div class="empty-state">
                <i class="fas fa-chart-bar"></i>
                <p>目标答案柱状图将在这里显示</p>
            </div>
        `
  }

  // 清除所有token的高亮状态
  document.querySelectorAll(".token.highlighted").forEach((token) => {
    token.classList.remove("highlighted")
  })

  // 清除所有token-with-visual的高亮状态
  document.querySelectorAll(".token-with-visual.highlighted").forEach((token) => {
    token.classList.remove("highlighted")
  })

  // 清除所有动态样式
  clearTokenAttentionColors("modelInputMirror")
  clearTokenAttentionColors("targetInputMirror")

  // 隐藏分数显示区域
  hideAllScoreDisplays()

  // 清除所有token分数显示容器
  const containers = document.querySelectorAll(".token-score-display-container")
  containers.forEach((el) => {
    if (el.parentNode) el.parentNode.removeChild(el)
  })

  // 清除全局范围信息
  if (window.allAttentionRanges) {
    delete window.allAttentionRanges
  }

  // 清理图表工具提示
  hideChartTooltip()

  analysisData = null
  selectedToken = null

  // 应用当前分析类型的布局
  updateAnalysisAreaVisibility(currentSubModule)
}

// Show loading state
function showLoading() {
  const loadingHtml = `
        <div class="loading">
            <div class="spinner"></div>
            <span>正在分析中，请稍候...</span>
        </div>
    `

  // 根据当前子模块加载不同区域
  if (currentSubModule === "attention-score") {
    document.getElementById("modelTokens").innerHTML = loadingHtml
    document.getElementById("targetTokens").innerHTML = loadingHtml
    document.getElementById("modelInputMirror").innerHTML = loadingHtml
    document.getElementById("targetInputMirror").innerHTML = loadingHtml
  } else {
    // 损失和熵分析在dynamicAnalysisArea的chart容器中显示加载状态
    document.getElementById("modelChart").innerHTML = loadingHtml
    document.getElementById("targetChart").innerHTML = loadingHtml
  }
}

// Display analysis results
function displayAnalysisResults(data) {
  analysisData = data

  // 根据当前分析类型渲染不同区域
  if (currentSubModule === "attention-score") {
    // Display model tokens
    document.getElementById("modelTokens").innerHTML = renderOutputTokens(data.original, "model")

    // Display target tokens
    if (data.target && data.target.length > 0) {
      document.getElementById("targetTokens").innerHTML = renderOutputTokens(data.target, "target")
    } else {
      document.getElementById("targetTokens").innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>目标答案数据不可用</p>
                    <small>API未返回目标答案的token化结果</small>
                </div>
            `
    }

    // 为两侧注意力高亮区域都渲染输入令牌
    document.getElementById("modelInputMirror").innerHTML = renderInputTokens(data.input)
    if (data.target && data.target.length > 0) {
      document.getElementById("targetInputMirror").innerHTML = renderInputTokens(data.input)
    } else {
      document.getElementById("targetInputMirror").innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>目标答案注意力数据不可用</p>
                </div>
            `
    }
  } else if (currentSubModule === "token-loss") {
    // 在dynamicAnalysisArea的chart容器中显示损失柱状图
    document.getElementById("modelChart").innerHTML = renderChart(data.original, "loss", "model")
    document.getElementById("targetChart").innerHTML = renderChart(data.target, "loss", "target")
  } else if (currentSubModule === "token-entropy") {
    // 在dynamicAnalysisArea的chart容器中显示熵柱状图
    document.getElementById("modelChart").innerHTML = renderChart(data.original, "entropy", "model")
    document.getElementById("targetChart").innerHTML = renderChart(data.target, "entropy", "target")
  }
}

// Render output tokens
function renderOutputTokens(tokens, answerType) {
  if (currentSubModule === "token-loss") {
    return renderTokensWithLoss(tokens, answerType)
  } else if (currentSubModule === "token-entropy") {
    return renderTokensWithEntropy(tokens, answerType)
  } else {
    return renderTokensWithAttention(tokens, answerType)
  }
}

// Render tokens with attention scores (original functionality)
function renderTokensWithAttention(tokens, answerType) {
  // 添加"All"按钮作为第一个元素
  let html = `<span class="token all-token" 
            data-answer-type="${answerType}"
            onclick="showAllAttention('${answerType}')"
            title="查看所有token的平均注意力分布">All</span>`

  // 添加其他tokens - 都使用原有的注意力高亮功能
  html += tokens
    .map((token) => {
      if (token.text.trim() === "") return token.text

      return `<span class="token" 
            data-token-id="${token.id}" 
            data-answer-type="${answerType}"
            data-attention="${token.attentionScore.toFixed(3)}"
            onclick="selectOutputToken(${token.id}, '${answerType}')"
            title="注意力分数: ${token.attentionScore.toFixed(3)} | 点击查看问题中对应的注意力高亮"
        >${token.text}</span>`
    })
    .join("")

  return html
}

// Render tokens with loss values (simplified for display only)
function renderTokensWithLoss(tokens, answerType) {
  const maxLoss = Math.max(...tokens.map((t) => t.lossValue))
  const minLoss = Math.min(...tokens.map((t) => t.lossValue))

  return tokens
    .map((token) => {
      if (token.text.trim() === "") return token.text

      // Determine loss level for styling
      const normalizedLoss = (token.lossValue - minLoss) / (maxLoss - minLoss)
      let lossClass = "loss-very-low"
      let lossLevel = "极低"
      let isHighValue = false

      if (normalizedLoss > 0.8) {
        lossClass = "loss-very-high"
        lossLevel = "极高"
        isHighValue = true
      } else if (normalizedLoss > 0.6) {
        lossClass = "loss-high"
        lossLevel = "高"
        isHighValue = true
      } else if (normalizedLoss > 0.4) {
        lossClass = "loss-medium"
        lossLevel = "中"
      } else if (normalizedLoss > 0.2) {
        lossClass = "loss-low"
        lossLevel = "低"
      } else {
        lossLevel = "极低"
      }

      // Calculate progress width (10% to 100% range)
      const progressWidth = Math.max(10, normalizedLoss * 100)

      return `<span class="token-with-visual loss-token ${lossClass} ${isHighValue ? "high-value" : ""}" 
            title="损失值: ${token.lossValue.toFixed(3)} | ${lossLevel}损失"
        >
            <div class="token-content">
                <div class="token-text">${token.text}</div>
                <div class="token-progress-container">
                    <div class="token-progress-bar">
                        <div class="token-progress-fill" style="width: ${progressWidth}%;"></div>
                    </div>
                    <div class="token-value-row">
                        <span class="token-value-display">${token.lossValue.toFixed(2)}</span>
                        <span class="token-level-badge">${lossLevel}</span>
                    </div>
                </div>
            </div>
        </span>`
    })
    .join("")
}

// Render tokens with entropy values (simplified for display only)
function renderTokensWithEntropy(tokens, answerType) {
  const maxEntropy = Math.max(...tokens.map((t) => t.entropyValue))
  const minEntropy = Math.min(...tokens.map((t) => t.entropyValue))

  return tokens
    .map((token) => {
      if (token.text.trim() === "") return token.text

      // Determine entropy level for styling
      const normalizedEntropy = (token.entropyValue - minEntropy) / (maxEntropy - minEntropy)
      let entropyClass = "entropy-very-low"
      let entropyLevel = "极低"
      let isHighValue = false

      if (normalizedEntropy > 0.8) {
        entropyClass = "entropy-very-high"
        entropyLevel = "极高"
        isHighValue = true
      } else if (normalizedEntropy > 0.6) {
        entropyClass = "entropy-high"
        entropyLevel = "高"
        isHighValue = true
      } else if (normalizedEntropy > 0.4) {
        entropyClass = "entropy-medium"
        entropyLevel = "中"
      } else if (normalizedEntropy > 0.2) {
        entropyClass = "entropy-low"
        entropyLevel = "低"
      } else {
        entropyLevel = "极低"
      }

      // Calculate progress width (10% to 100% range)
      const progressWidth = Math.max(10, normalizedEntropy * 100)

      return `<span class="token-with-visual entropy-token ${entropyClass} ${isHighValue ? "high-value" : ""}" 
            title="熵值: ${token.entropyValue.toFixed(3)} | ${entropyLevel}熵值"
        >
            <div class="token-content">
                <div class="token-text">${token.text}</div>
                <div class="token-progress-container">
                    <div class="token-progress-bar">
                        <div class="token-progress-fill" style="width: ${progressWidth}%;"></div>
                    </div>
                    <div class="token-value-row">
                        <span class="token-value-display">${token.entropyValue.toFixed(2)}</span>
                        <span class="token-level-badge">${entropyLevel}</span>
                    </div>
                </div>
            </div>
        </span>`
    })
    .join("")
}

// Render input tokens
function renderInputTokens(tokens) {
  return tokens
    .map((token, index) => {
      if (token.trim() === "") return token

      return `<span class="token" 
            data-input-id="${index}"
            id="input-token-${index}"
            onclick="showInputTokenPopupScore(${index}, event)"
            style="cursor: pointer;"
        >${token}</span>`
    })
    .join("")
}

// 存储chart数据以便范围选择使用
window.chartData = window.chartData || {}

// Render chart for loss or entropy analysis - 简洁版本
function renderChart(tokens, analysisType, answerType) {
  const validTokens = tokens.filter((token) => token.text.trim() !== "")
  if (validTokens.length === 0) {
    return `
            <div class="empty-state">
                <i class="fas fa-chart-bar"></i>
                <p>没有可分析的token</p>
            </div>
        `
  }

  // 存储数据供范围选择使用
  const chartKey = `${answerType}-${analysisType}`
  window.chartData[chartKey] = validTokens

  // 获取数值和计算范围
  let values, maxValue, minValue
  if (analysisType === "loss") {
    values = validTokens.map((t) => t.lossValue)
  } else {
    values = validTokens.map((t) => t.entropyValue)
  }

  maxValue = Math.max(...values)
  minValue = Math.min(...values)
  const avgValue = values.reduce((sum, v) => sum + v, 0) / values.length

  const chartClass = analysisType === "loss" ? "loss-chart" : "entropy-chart"

  // 保持原始顺序，不排序
  const tokensToRender = validTokens

  const renderTokenItems = (minValueFilter = null, maxValueFilter = null) => {
    // 如果没有指定过滤范围，使用全部token
    let tokensToFilter = tokensToRender
    if (minValueFilter !== null && maxValueFilter !== null) {
      tokensToFilter = tokensToRender.filter(token => {
        const value = analysisType === "loss" ? token.lossValue : token.entropyValue
        return value >= minValueFilter && value <= maxValueFilter
      })
    }
    
    return tokensToFilter
      .map((token, index) => {
        const value = analysisType === "loss" ? token.lossValue : token.entropyValue
        const normalizedValue = maxValue > minValue ? (value - minValue) / (maxValue - minValue) : 0
        
        // 找到原始token在全部token中的位置
        const originalIndex = tokensToRender.findIndex(t => t === token)

        // 确定级别样式
        let levelClass = `${analysisType}-very-low`
        if (normalizedValue > 0.8) {
          levelClass = `${analysisType}-very-high`
        } else if (normalizedValue > 0.6) {
          levelClass = `${analysisType}-high`
        } else if (normalizedValue > 0.4) {
          levelClass = `${analysisType}-medium`
        } else if (normalizedValue > 0.2) {
          levelClass = `${analysisType}-low`
        }

        // 计算柱状图宽度 - 值接近0时不显示条
        let barWidth
        if (value < 0.0005) {
          // 显示为0.000的值不显示条
          barWidth = 0
        } else {
          barWidth = Math.max(2, 5 + normalizedValue * 90) // 非零值最小2%，正常5%-95%范围
        }

        return `
                <div class="chart-item" 
                     onmouseenter="showChartTooltip(event, '${token.text.replace(/'/g, "\\'")}', ${value}, '${analysisType === "loss" ? "损失值" : "熵值"}')"
                     onmouseleave="hideChartTooltip()">
                        <div class="chart-token">${token.text}</div>
                            <div class="chart-bar ${levelClass}" style="width: ${barWidth}%;"></div>
                    <div class="chart-value">${value.toFixed(3)}</div>
                    </div>
                `
      })
      .join("")
  }

  // 存储渲染函数供范围更新使用
  window.chartData[chartKey + '_renderer'] = renderTokenItems

  return `
        <div class="chart-container ${chartClass}">
            <div class="chart-header">
                <div class="chart-title">
                    <i class="fas fa-chart-bar"></i>
                    ${analysisType === "loss" ? "损失值" : "熵值"}分析 (${answerType === "model" ? "模型答案" : "目标答案"})
                </div>
                <div class="chart-summary">
                    <span><i class="fas fa-list"></i> ${validTokens.length} 个token</span>
                    <span><i class="fas fa-arrow-up"></i> 最大: ${maxValue.toFixed(3)}</span>
                    <span><i class="fas fa-arrow-down"></i> 最小: ${minValue.toFixed(3)}</span>
                    <span><i class="fas fa-calculator"></i> 平均: ${avgValue.toFixed(3)}</span>
                </div>
            </div>
            <div class="chart-range-selector">
                <div class="range-selector-label">
                    <i class="fas fa-filter"></i>
                    ${analysisType === "loss" ? "损失值" : "熵值"}范围:
                </div>
                <div class="range-inputs">
                    <input type="number" id="value-min-${answerType}-${analysisType}" 
                           min="${minValue.toFixed(3)}" max="${maxValue.toFixed(3)}" 
                           step="0.001" value="${minValue.toFixed(3)}" 
                           onchange="updateChartValueRange('${answerType}', '${analysisType}')">
                    <span class="range-separator">-</span>
                    <input type="number" id="value-max-${answerType}-${analysisType}" 
                           min="${minValue.toFixed(3)}" max="${maxValue.toFixed(3)}" 
                           step="0.001" value="${maxValue.toFixed(3)}" 
                           onchange="updateChartValueRange('${answerType}', '${analysisType}')">
                    <button class="range-reset-btn" onclick="resetChartValueRange('${answerType}', '${analysisType}')">
                        <i class="fas fa-refresh"></i> 重置
                    </button>
                </div>
            </div>
            <div class="chart-content" id="chart-content-${answerType}-${analysisType}">
                ${renderTokenItems()}
            </div>
            <div class="chart-resize-handle" onmousedown="startChartResize(event, 'chart-content-${answerType}-${analysisType}')"></div>
        </div>
    `
}

// 更新图表数值范围显示
function updateChartValueRange(answerType, analysisType) {
  const chartKey = `${answerType}-${analysisType}`
  const minInput = document.getElementById(`value-min-${chartKey}`)
  const maxInput = document.getElementById(`value-max-${chartKey}`)
  const chartContent = document.getElementById(`chart-content-${chartKey}`)
  
  if (!minInput || !maxInput || !chartContent || !window.chartData[chartKey]) {
    return
  }
  
  const allTokens = window.chartData[chartKey]
  let minValue = parseFloat(minInput.value)
  let maxValue = parseFloat(maxInput.value)
  
  // 获取所有值的范围用于验证
  const allValues = allTokens.map(token => 
    analysisType === "loss" ? token.lossValue : token.entropyValue
  )
  const absoluteMin = Math.min(...allValues)
  const absoluteMax = Math.max(...allValues)
  
  // 验证范围
  if (minValue < absoluteMin) minValue = absoluteMin
  if (maxValue > absoluteMax) maxValue = absoluteMax
  if (minValue >= maxValue) {
    minValue = Math.max(absoluteMin, maxValue - 0.001)
  }
  
  // 更新输入框显示
  minInput.value = minValue.toFixed(3)
  maxInput.value = maxValue.toFixed(3)
  
  // 重新渲染图表内容
  const renderer = window.chartData[chartKey + '_renderer']
  if (renderer) {
    chartContent.innerHTML = renderer(minValue, maxValue)
  }
  
  // 更新统计信息
  updateChartValueSummary(answerType, analysisType, minValue, maxValue)
}

// 更新图表数值统计信息
function updateChartValueSummary(answerType, analysisType, minValueFilter, maxValueFilter) {
  const chartKey = `${answerType}-${analysisType}`
  const allTokens = window.chartData[chartKey]
  
  if (!allTokens) return
  
  // 筛选符合范围的token
  const filteredTokens = allTokens.filter(token => {
    const value = analysisType === "loss" ? token.lossValue : token.entropyValue
    return value >= minValueFilter && value <= maxValueFilter
  })
  
  if (filteredTokens.length === 0) return
  
  const values = filteredTokens.map(token => 
    analysisType === "loss" ? token.lossValue : token.entropyValue
  )
  
  const maxValue = Math.max(...values)
  const minValue = Math.min(...values)
  const avgValue = values.reduce((sum, v) => sum + v, 0) / values.length
  
  // 查找并更新统计显示
  const chartContainer = document.getElementById(`chart-content-${chartKey}`)?.closest('.chart-container')
  if (chartContainer) {
    const summary = chartContainer.querySelector('.chart-summary')
    if (summary) {
      summary.innerHTML = `
        <span><i class="fas fa-list"></i> ${filteredTokens.length} 个token (${analysisType === "loss" ? "损失值" : "熵值"}: ${minValueFilter.toFixed(3)}-${maxValueFilter.toFixed(3)})</span>
        <span><i class="fas fa-arrow-up"></i> 最大: ${maxValue.toFixed(3)}</span>
        <span><i class="fas fa-arrow-down"></i> 最小: ${minValue.toFixed(3)}</span>
        <span><i class="fas fa-calculator"></i> 平均: ${avgValue.toFixed(3)}</span>
      `
    }
  }
}

// 重置图表数值范围
function resetChartValueRange(answerType, analysisType) {
  const chartKey = `${answerType}-${analysisType}`
  const minInput = document.getElementById(`value-min-${chartKey}`)
  const maxInput = document.getElementById(`value-max-${chartKey}`)
  
  if (!minInput || !maxInput || !window.chartData[chartKey]) {
    return
  }
  
  const allTokens = window.chartData[chartKey]
  const allValues = allTokens.map(token => 
    analysisType === "loss" ? token.lossValue : token.entropyValue
  )
  const absoluteMin = Math.min(...allValues)
  const absoluteMax = Math.max(...allValues)
  
  minInput.value = absoluteMin.toFixed(3)
  maxInput.value = absoluteMax.toFixed(3)
  
  updateChartValueRange(answerType, analysisType)
}

// Select output token and highlight corresponding input
function selectOutputToken(tokenId, answerType) {
  if (!analysisData) return

  // 只隐藏对应侧的分数显示区域
  const displayAreaId = answerType === "model" ? "modelScoreDisplay" : "targetScoreDisplay"
  hideScoreDisplay(displayAreaId)

  // 清除全局范围信息（因为不再是All模式）
  if (window.allAttentionRanges) {
    delete window.allAttentionRanges
  }

  // 清除所有 All token 的高亮状态
  document.querySelectorAll(`.all-token[data-answer-type="${answerType}"]`).forEach((token) => {
    token.classList.remove("highlighted")
  })

  // 只清除当前答案类型的token高亮
  document.querySelectorAll(`.token[data-answer-type="${answerType}"]`).forEach((token) => {
    if (!token.classList.contains("all-token")) {
      token.classList.remove("highlighted")
    }
  })

  // 清除对应侧的输入区域高亮
  const inputMirrorId = answerType === "model" ? "modelInputMirror" : "targetInputMirror"
  clearTokenAttentionColors(inputMirrorId)

  // Highlight selected output token
  const selectedElement = document.querySelector(`[data-token-id="${tokenId}"][data-answer-type="${answerType}"]`)
  if (selectedElement) {
    selectedElement.classList.add("highlighted")

    // Get attention scores for input tokens
    const currentData = answerType === "model" ? analysisData.original : analysisData.target
    const tokenData = currentData.find((t) => t.id === tokenId)

    if (tokenData && tokenData.inputAttention) {
      // 计算当前token的注意力分数范围
      const validScores = tokenData.inputAttention.filter((_, index) => analysisData.input[index].trim() !== "")

      // 处理边界情况
      if (validScores.length === 0) {
        console.warn("没有有效的注意力分数")
        return
      }

      const maxScore = Math.max(...validScores)
      const minScore = Math.min(...validScores)

      console.log(`Token "${tokenData.text}" 注意力分数范围: ${minScore.toFixed(3)} - ${maxScore.toFixed(3)}`)

      // Apply attention-based highlighting to input tokens
      tokenData.inputAttention.forEach((score, inputIndex) => {
        // 只高亮对应侧的输入区域
        const inputElement = document.querySelector(`#${inputMirrorId} [data-input-id="${inputIndex}"]`)

        if (inputElement && analysisData.input[inputIndex].trim()) {
          // 使用统一的颜色设置函数确保一致性
          setTokenAttentionColor(
            inputElement,
            score,
            minScore,
            maxScore,
            `注意力分数: ${score.toFixed(3)} (范围: ${minScore.toFixed(3)}-${maxScore.toFixed(3)}) | 对应${answerType === "model" ? "模型" : "目标"}答案token: "${tokenData.text.trim()}"`,
          )
        }
      })
    }
  }
}

// 显示某个答案的所有注意力分布
function showAllAttention(answerType) {
  if (!analysisData) return

  // 只隐藏对应侧的分数显示区域
  const displayAreaId = answerType === "model" ? "modelScoreDisplay" : "targetScoreDisplay"
  hideScoreDisplay(displayAreaId)

  // 只清除当前答案类型的token高亮（保留其他侧的高亮）
  document.querySelectorAll(`.token[data-answer-type="${answerType}"]`).forEach((token) => {
    if (!token.classList.contains("all-token")) {
      token.classList.remove("highlighted")
    }
  })

  // 激活当前All token，将它设为高亮
  const allToken = document.querySelector(`.all-token[data-answer-type="${answerType}"]`)
  if (allToken) {
    allToken.classList.add("highlighted")
  }

  // 清除对应侧的输入区域高亮
  const inputMirrorId = answerType === "model" ? "modelInputMirror" : "targetInputMirror"
  clearTokenAttentionColors(inputMirrorId)

  // 获取当前数据
  const currentData = answerType === "model" ? analysisData.original : analysisData.target
  const inputTokens = analysisData.input

  // 计算每个输入token的平均注意力分数
  const avgAttentionScores = []

  // 初始化平均分数数组
  for (let i = 0; i < inputTokens.length; i++) {
    avgAttentionScores[i] = 0
  }

  // 累加所有token对应的注意力分数
  let validTokenCount = 0
  currentData.forEach((token) => {
    if (token.text.trim() !== "" && token.inputAttention) {
      validTokenCount++
      token.inputAttention.forEach((score, idx) => {
        avgAttentionScores[idx] += score
      })
    }
  })

  // 计算平均值并存储全局范围信息
  if (validTokenCount > 0) {
    for (let i = 0; i < avgAttentionScores.length; i++) {
      avgAttentionScores[i] = avgAttentionScores[i] / validTokenCount
    }

    // 计算所有平均分数的全局范围
    const validAvgScores = avgAttentionScores.filter((_, index) => inputTokens[index].trim() !== "")

    if (validAvgScores.length === 0) {
      console.warn("没有有效的平均注意力分数")
      return
    }

    const maxAvgScore = Math.max(...validAvgScores)
    const minAvgScore = Math.min(...validAvgScores)

    console.log(`All tokens 平均注意力分数范围: ${minAvgScore.toFixed(3)} - ${maxAvgScore.toFixed(3)}`)

    // 存储全局范围信息，供弹窗使用
    if (!window.allAttentionRanges) {
      window.allAttentionRanges = {}
    }
    window.allAttentionRanges[answerType] = {
      min: minAvgScore,
      max: maxAvgScore,
      scores: avgAttentionScores,
    }

    // 应用平均注意力分数高亮
    avgAttentionScores.forEach((score, inputIndex) => {
      const inputElement = document.querySelector(`#${inputMirrorId} [data-input-id="${inputIndex}"]`)

      if (inputElement && inputTokens[inputIndex].trim()) {
        // 使用统一的颜色设置函数确保一致性
        setTokenAttentionColor(
          inputElement,
          score,
          minAvgScore,
          maxAvgScore,
          `平均注意力分数: ${score.toFixed(3)} (全局范围: ${minAvgScore.toFixed(3)}-${maxAvgScore.toFixed(3)})`,
        )
      }
    })
  }
}

// Calculate average attention score
function calculateAverageAttention(tokens) {
  const validTokens = tokens.filter((t) => t.text.trim() !== "")
  return validTokens.reduce((sum, token) => sum + token.attentionScore, 0) / validTokens.length
}

// 根据动态范围生成注意力颜色
function getAttentionColorByRange(currentScore, minScore, maxScore) {
  // 如果最大值和最小值相同，使用中等强度
  if (maxScore === minScore) {
    console.log(`所有分数相同: ${currentScore.toFixed(3)}, 使用中等强度`)
    return getAttentionColorByIntensity(0.5, currentScore)
  }

  // 计算当前分数在范围内的相对位置 (0-1)
  const relativeIntensity = (currentScore - minScore) / (maxScore - minScore)

  // 添加调试信息
  if (Math.random() < 0.1) {
    // 只输出10%的调试信息，避免过多日志
    console.log(
      `分数映射: ${currentScore.toFixed(3)} 在范围 [${minScore.toFixed(3)}, ${maxScore.toFixed(3)}] 中的强度为 ${relativeIntensity.toFixed(3)}`,
    )
  }

  return getAttentionColorByIntensity(relativeIntensity, currentScore)
}

// 统一的清除token注意力颜色函数
function clearTokenAttentionColors(inputMirrorId) {
  document.querySelectorAll(`#${inputMirrorId} .token[data-input-id]`).forEach((token) => {
    token.className = "token"
    token.title = ""
    token.style.backgroundColor = ""
    token.style.color = ""
    token.style.fontWeight = ""
    token.style.borderRadius = ""
    token.style.padding = ""
    token.style.cursor = ""
    token.style.transition = ""
    
    // 清除存储的分数信息
    token.removeAttribute('data-current-score')
    token.removeAttribute('data-current-min-score')
    token.removeAttribute('data-current-max-score')
    token.removeAttribute('data-current-title')
  })
}

// 统一的token注意力颜色设置函数，确保两个区域颜色风格一致
function setTokenAttentionColor(inputElement, score, minScore, maxScore, titleText) {
  // 基于分数范围动态设置颜色
  const backgroundColor = getAttentionColorByRange(score, minScore, maxScore)
  const relativeIntensity = maxScore > minScore ? (score - minScore) / (maxScore - minScore) : 0.5

  // 固定使用深色文字，不根据背景透明度改变
  const alpha = 0.01 + relativeIntensity * (0.95 - 0.01)
  const textColor = "#1e293b" // 始终使用深色文字

  // 统一设置样式
  inputElement.className = "token attention-highlight"
  inputElement.style.backgroundColor = backgroundColor
  inputElement.style.color = textColor
  inputElement.style.transition = "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
  inputElement.style.fontWeight = "500"
  inputElement.style.borderRadius = "0.25rem"
  inputElement.style.padding = "0.125rem 0.25rem"
  inputElement.title = titleText

  // 添加统一的悬停效果
  inputElement.style.cursor = "pointer"
  
  // 存储当前显示的分数信息到data属性中，用于点击时准确显示
  inputElement.setAttribute('data-current-score', score)
  inputElement.setAttribute('data-current-min-score', minScore)
  inputElement.setAttribute('data-current-max-score', maxScore)
  inputElement.setAttribute('data-current-title', titleText)
}

// 根据相对强度生成颜色，使用青色的不同透明度
function getAttentionColorByIntensity(intensity, originalScore) {
  // 确保强度在0-1之间
  intensity = Math.max(0, Math.min(1, intensity))

  // 使用青色，与示例颜色一致
  const baseColor = [2, 169, 169] // rgb(2, 169, 169) (青色)

  // 根据强度计算透明度，提供极大的对比度范围
  // 最小透明度0.01（几乎完全看不见），最大透明度0.95（非常深）
  const minAlpha = 0.01
  const maxAlpha = 0.95
  const alpha = minAlpha + intensity * (maxAlpha - minAlpha)

  // 所有强度级别都使用相同的青色，只改变透明度
  const colorString = `rgba(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]}, ${alpha})`

  // 添加调试信息（偶尔输出）
  if (Math.random() < 0.05) {
    // 5%的概率输出调试信息
    console.log(`颜色生成: 强度=${intensity.toFixed(3)}, 透明度=${alpha.toFixed(3)}, 颜色=${colorString}`)
  }

  return colorString
}

// 保留原有函数以兼容其他可能的调用
function getAttentionColor(score) {
  return getAttentionColorByIntensity(score, score)
}

// 在内联区域显示分数信息
function showInlineScoreDisplay(displayAreaId, tokenText, scoreInfo) {
  const displayArea = document.getElementById(displayAreaId)
  if (!displayArea) return

  // 显示分数区域
  displayArea.style.display = "block"

  // 获取内容区域
  const contentArea = displayArea.querySelector(".score-display-content")

  // 生成分数内容
  let content = ""
  if (scoreInfo.scores && scoreInfo.scores.length > 0) {
    const scoreData = scoreInfo.scores[0] // 取第一个分数数据

    // 计算百分比（如果有范围信息）
    let percentage = ""
    if (scoreData.range && scoreData.range.min !== undefined && scoreData.range.max !== undefined) {
      const range = scoreData.range.max - scoreData.range.min
      if (range > 0) {
        const percent = (((scoreData.score - scoreData.range.min) / range) * 100).toFixed(1)
        percentage = `${percent}%`
      }
    }

    content = `
            <div class="score-item">
                <div class="score-token-info">
                    <div class="score-token-text">"${tokenText}"</div>
                    <div class="score-token-value">${scoreData.score.toFixed(6)}</div>
                </div>
                ${
                  percentage
                    ? `
                <div class="score-progress-container">
                    <div class="score-progress-header">
                        <div class="score-progress-label">相对强度</div>
                        <div class="score-progress-percentage">${percentage}</div>
                    </div>
                    <div class="score-progress-bar">
                        <div class="score-progress-fill" style="width: ${percentage};"></div>
                    </div>
                </div>
                `
                    : ""
                }
                <div class="score-details">
                    <div class="score-range-info">
                        <div class="score-range-item">
                            <div class="score-range-label">最小值</div>
                            <div class="score-range-value">${scoreData.range ? scoreData.range.min.toFixed(6) : "N/A"}</div>
                        </div>
                        <div class="score-range-item">
                            <div class="score-range-label">最大值</div>
                            <div class="score-range-value">${scoreData.range ? scoreData.range.max.toFixed(6) : "N/A"}</div>
                        </div>
                    </div>
                </div>
            </div>
        `
  } else {
    content = `
            <div class="score-item">
                <div class="score-token-info">
                    <div class="score-token-text">"${tokenText}"</div>
                    <div class="score-token-value">无数据</div>
                </div>
            </div>
        `
  }

  contentArea.innerHTML = content
}

// 隐藏分数显示区域
function hideScoreDisplay(displayAreaId) {
  const displayArea = document.getElementById(displayAreaId)
  if (displayArea) {
    displayArea.style.display = "none"
  }
}

// 隐藏所有分数显示区域
function hideAllScoreDisplays() {
  hideScoreDisplay("modelScoreDisplay")
  hideScoreDisplay("targetScoreDisplay")
}

// 显示输入token的弹出分数（compression页面风格）
function showInputTokenPopupScore(inputTokenIndex, event) {
  console.log("点击了输入token:", inputTokenIndex)

  if (!analysisData) {
    console.warn("没有分析数据")
    return
  }

  const inputToken = analysisData.input[inputTokenIndex]
  if (!inputToken || !inputToken.trim()) {
    console.warn("输入token为空")
    return
  }

  // 确定当前点击的token属于哪个区域
  const tokenElement = event.target
  const isModelArea = tokenElement.closest("#modelInputMirror") !== null
  const isTargetArea = tokenElement.closest("#targetInputMirror") !== null

  console.log("输入token内容:", inputToken, "区域:", isModelArea ? "模型" : isTargetArea ? "目标" : "未知")

  // 检查token是否有存储的分数信息（即是否已经被高亮）
  const currentScore = tokenElement.getAttribute('data-current-score')
  const currentMinScore = tokenElement.getAttribute('data-current-min-score')
  const currentMaxScore = tokenElement.getAttribute('data-current-max-score')
  const currentTitle = tokenElement.getAttribute('data-current-title')

  if (currentScore && currentMinScore && currentMaxScore) {
    // 直接使用存储的分数信息，确保与颜色完全匹配
    const scoreData = {
      score: parseFloat(currentScore),
      range: {
        min: parseFloat(currentMinScore),
        max: parseFloat(currentMaxScore)
      }
    }
    
    console.log("使用存储的分数信息:", scoreData)
    showCompressionStyleScore(tokenElement, inputToken, scoreData)
    return
  }

  // 如果没有存储的分数信息，说明token没有被高亮，提示用户先选择output token
  const areaName = isModelArea ? "左侧模型答案" : "右侧目标答案"
  console.warn(`Token没有被高亮，可能${areaName}区域没有激活的输出token`)
  alert(`请先点击${areaName}中的某个token（或All按钮），然后再点击下方问题中的token查看分数`)
}

// 收集输入token的分数信息
function collectTokenScoreInfo(inputTokenIndex, areaType) {
  if (!analysisData) {
    console.warn("没有分析数据")
    return null
  }

  const inputToken = analysisData.input[inputTokenIndex]
  if (!inputToken || !inputToken.trim()) {
    console.warn("输入token为空")
    return null
  }

  // 获取当前区域的数据
  const currentData = areaType === "model" ? analysisData.original : analysisData.target

  // 查找当前激活的output token
  const activeTokens = []
  
  // 检查是否All token激活
  const allToken = document.querySelector(`#${areaType}Tokens .all-token[data-answer-type="${areaType}"]`)
  const isAllTokenActive = allToken && allToken.classList.contains('highlighted')
  
  // 查找单独激活的token
  let singleActiveToken = null
  if (!isAllTokenActive) {
    // 如果不是All模式，查找单独激活的token
    currentData.forEach((tokenData) => {
      const outputTokenElement = document.querySelector(`#${areaType}Tokens [data-token-id="${tokenData.id}"]`)
      if (outputTokenElement && outputTokenElement.classList.contains('highlighted')) {
        singleActiveToken = tokenData
      }
    })
  }
  
  // 如果是单个token激活，优先显示该token的分数和范围
  if (singleActiveToken && singleActiveToken.inputAttention && singleActiveToken.inputAttention[inputTokenIndex] !== undefined) {
    const score = singleActiveToken.inputAttention[inputTokenIndex]
    const validScores = singleActiveToken.inputAttention.filter((_, index) => analysisData.input[index].trim() !== "")
    
    activeTokens.push({
      label: "注意力分数",
      outputToken: singleActiveToken.text.trim(),
      score: score,
      tokenId: singleActiveToken.id,
      range: {
        min: Math.min(...validScores),
        max: Math.max(...validScores)
      }
    })
  } else {
    // All模式或没有找到单个激活token时，收集所有相关分数
    currentData.forEach((tokenData) => {
      if (tokenData.inputAttention && tokenData.inputAttention[inputTokenIndex] !== undefined) {
        const score = tokenData.inputAttention[inputTokenIndex]
        
        // 检查这个token是否当前激活（有高亮显示）或者All token激活
        const outputTokenElement = document.querySelector(`#${areaType}Tokens [data-token-id="${tokenData.id}"]`)
        const isTokenActive = outputTokenElement && outputTokenElement.classList.contains('highlighted')
        
        if (isTokenActive || isAllTokenActive) {
          activeTokens.push({
            label: "注意力分数",
            outputToken: tokenData.text.trim(),
            score: score,
            tokenId: tokenData.id,
            range: {
              min: Math.min(...tokenData.inputAttention.filter((_, index) => analysisData.input[index].trim() !== "")),
              max: Math.max(...tokenData.inputAttention.filter((_, index) => analysisData.input[index].trim() !== ""))
            }
          })
        }
      }
    })
  }

  if (activeTokens.length === 0) {
    console.warn("没有找到激活的output token")
    return null
  }

  return {
    inputToken: inputToken,
    inputTokenIndex: inputTokenIndex,
    scores: activeTokens
  }
}

// 显示输入token的注意力分数（原有功能，保持不变用于目标答案区域）
function showInputTokenScore(inputTokenIndex, event) {
    console.log('点击了输入token:', inputTokenIndex);
    
    if (!analysisData) {
        console.warn('没有分析数据');
        return;
    }

    const inputToken = analysisData.input[inputTokenIndex];
    if (!inputToken || !inputToken.trim()) {
        console.warn('输入token为空');
        return;
    }

    // 确定当前点击的token属于哪个区域
    const tokenElement = event.target;
    const isModelArea = tokenElement.closest('#modelInputMirror') !== null;
    const isTargetArea = tokenElement.closest('#targetInputMirror') !== null;
    
    console.log('输入token内容:', inputToken, '区域:', isModelArea ? '模型' : (isTargetArea ? '目标' : '未知'));

    // 收集当前高亮状态下的分数信息，限制在对应区域
    const scoreInfo = collectTokenScoreInfo(inputTokenIndex, isModelArea ? 'model' : 'target');
    console.log('收集到的分数信息:', scoreInfo);
    
    if (!scoreInfo || scoreInfo.scores.length === 0) {
        const areaName = isModelArea ? '左侧模型答案' : '右侧目标答案'
        console.warn(`没有收集到分数信息，可能${areaName}区域没有激活的输出token`)
        alert(`请先点击${areaName}中的某个token（或All按钮），然后再点击下方问题中的token查看分数`)
        return
    }

    // 使用compression页面风格的弹出分数显示
    showCompressionStyleScore(tokenElement, inputToken, scoreInfo.scores[0])
}

// 显示compression页面风格的token分数
function showCompressionStyleScore(tokenElement, tokenText, scoreData) {
  // 移除所有现有的分数显示
  const existingDisplays = document.querySelectorAll('.token-score-display-container')
  existingDisplays.forEach(el => {
    if (el.parentNode) {
      el.parentNode.removeChild(el)
      if (el._cleanup) {
        el._cleanup()
      }
    }
  })

  // 根据分数范围计算百分比
  const scoreRange = scoreData.range.max - scoreData.range.min
  let scorePercentage = 0
  if (scoreRange > 0) {
    scorePercentage = ((scoreData.score - scoreData.range.min) / scoreRange) * 100
  } else {
    scorePercentage = 50 // 如果所有分数相同，使用50%
  }

  // 根据分数百分比确定注意力等级
  let attentionLevel = "极低"
  let attentionColor = "#02a9a9" // 使用青色主题
  if (scorePercentage >= 80) {
    attentionLevel = "极高"
  } else if (scorePercentage >= 60) {
    attentionLevel = "高"
  } else if (scorePercentage >= 40) {
    attentionLevel = "中等"
  } else if (scorePercentage >= 20) {
    attentionLevel = "低"
  }

  // 获取token的位置信息
  const tokenRect = tokenElement.getBoundingClientRect()
  
  // 创建分数显示容器
  const container = document.createElement('div')
  container.className = 'token-score-display-container'
  
  // 设置容器样式 - 使用absolute定位相对于页面，并添加滚动监听
  container.style.cssText = `
    position: absolute !important;
    z-index: 2147483647 !important;
    pointer-events: none !important;
    left: ${tokenRect.left + window.scrollX + tokenRect.width/2}px !important;
    top: ${tokenRect.bottom + window.scrollY}px !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 1 !important;
    visibility: visible !important;
  `
  
  // 创建分数显示元素
  const scoreDisplay = document.createElement('div')
  scoreDisplay.className = 'token-score-display'
  
  // 设置分数显示的内容
  const contentDiv = document.createElement('div')
  contentDiv.style.cssText = 'display: flex; align-items: center; gap: 0.5rem;'
  
  const scoreText = document.createElement('span')
  // 显示原始分数、百分比和分数范围，与compression页面格式一致
  scoreText.textContent = `分数: ${scoreData.score.toFixed(4)} (${scorePercentage.toFixed(1)}%) | 范围: ${scoreData.range.min.toFixed(4)} - ${scoreData.range.max.toFixed(4)}`
  scoreText.style.cssText = 'color: #374151; font-weight: 500; font-size: 0.75rem;'
  
  const levelBadge = document.createElement('span')
  levelBadge.textContent = attentionLevel
  levelBadge.style.cssText = `
    background: ${attentionColor}; 
    color: white; 
    padding: 0.125rem 0.375rem; 
    border-radius: 1rem; 
    font-size: 0.7rem; 
    font-weight: 370;
  `
  
  // 添加指向token的箭头
  const arrow = document.createElement('div')
  arrow.style.cssText = `
    position: absolute !important;
    top: -6px !important;
    left: 50% !important;
    margin-left: -5px !important;
    width: 10px !important;
    height: 10px !important;
    background-color: white !important;
    border-left: 2px solid ${attentionColor} !important;
    border-top: 2px solid ${attentionColor} !important;
    transform: rotate(45deg) !important;
    z-index: 2147483647 !important;
  `
  
  // 组装所有元素
  contentDiv.appendChild(scoreText)
  contentDiv.appendChild(levelBadge)
  scoreDisplay.appendChild(contentDiv)
  scoreDisplay.appendChild(arrow)
  
  // 设置分数显示的样式
  scoreDisplay.style.cssText = `
    position: absolute !important;
    z-index: 2147483647 !important;
    background-color: white !important;
    border: 2px solid ${attentionColor} !important;
    border-radius: 0.5rem !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    padding: 0.5rem 0.75rem !important;
    white-space: nowrap !important;
    top: 5px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    opacity: 1 !important;
    visibility: visible !important;
    pointer-events: none !important;
  `
  
  // 将分数显示添加到容器
  container.appendChild(scoreDisplay)
  
  // 将容器添加到body
  document.body.appendChild(container)
  
  // 添加滚动监听，让弹窗跟随token位置
  const updatePosition = () => {
    const newTokenRect = tokenElement.getBoundingClientRect()
    container.style.left = `${newTokenRect.left + window.scrollX + newTokenRect.width/2}px`
    container.style.top = `${newTokenRect.bottom + window.scrollY}px`
  }
  
  // 监听页面滚动和窗口大小变化
  window.addEventListener('scroll', updatePosition)
  window.addEventListener('resize', updatePosition)
  
  // 3秒后自动移除，并清理监听器
  const removeTimeout = setTimeout(() => {
    if (container.parentNode) {
      container.parentNode.removeChild(container)
      window.removeEventListener('scroll', updatePosition)
      window.removeEventListener('resize', updatePosition)
    }
  }, 3000)
  
  // 存储清理函数到容器，以便手动移除时也能清理
  container._cleanup = () => {
    clearTimeout(removeTimeout)
    window.removeEventListener('scroll', updatePosition)
    window.removeEventListener('resize', updatePosition)
  }
  
  // 点击其他位置时移除分数显示
  const removeScoreDisplay = function(event) {
    if (event.target !== tokenElement && !tokenElement.contains(event.target)) {
      if (container.parentNode) {
        container.parentNode.removeChild(container)
        if (container._cleanup) {
          container._cleanup()
        }
      }
      document.removeEventListener('click', removeScoreDisplay)
    }
  }
  document.addEventListener('click', removeScoreDisplay)
  
  // 按ESC键移除分数显示
  const handleEscape = function(event) {
    if (event.key === 'Escape') {
      if (container.parentNode) {
        container.parentNode.removeChild(container)
        if (container._cleanup) {
          container._cleanup()
        }
      }
      document.removeEventListener('keydown', handleEscape)
    }
  }
  document.addEventListener('keydown', handleEscape)
}

// 加载输入示例
function loadInputSample() {
  const inputSample = `请解释一下什么是大语言模型，以及它们是如何工作的？
请用通俗易懂的语言回答，适合非技术人员理解。`
  document.getElementById('inputContent').value = inputSample
}

// 加载目标示例
function loadTargetSample() {
  const targetSample = `大语言模型是一种人工智能系统，它通过学习大量的文本数据来理解和生成人类语言。想象一下，它就像是一个"超级记事本"，记住了互联网上的大量文章、书籍和对话。`
  document.getElementById('targetContent').value = targetSample
}

// 复制模型生成答案
function copyModelAnswer() {
  if (!analysisData || !analysisData.modelAnswer) {
    showNotification('没有可复制的内容', 'error')
    return
  }

  const textToCopy = analysisData.modelAnswer

  if (navigator.clipboard) {
    navigator.clipboard.writeText(textToCopy)
      .then(() => showCopySuccess())
      .catch(err => {
        console.error('复制失败:', err)
        // 如果现代API失败，回退到兼容模式
        fallbackCopy(textToCopy)
      })
  } else {
    // 回退到兼容模式
    fallbackCopy(textToCopy)
  }
}

// 回退到兼容模式的复制方法
function fallbackCopy(text) {
  // 创建一个临时textarea元素来复制文本
  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.style.position = 'fixed'  // 确保在屏幕外
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  textarea.select()

  try {
    // 执行复制命令
    const successful = document.execCommand('copy')
    if (successful) {
      showCopySuccess()
    } else {
      alert('复制失败，请手动复制内容')
    }
  } catch (err) {
    console.error('复制失败:', err)
    alert('复制失败，请手动复制内容')
  }

  // 移除临时元素
  document.body.removeChild(textarea)
}

// 显示复制成功的视觉反馈
function showCopySuccess() {
  // 显示复制成功的反馈
  const copyBtns = document.querySelectorAll('.copy-btn')
  copyBtns.forEach(btn => {
    const originalIcon = btn.innerHTML
    btn.innerHTML = '<i class="fas fa-check"></i>'
    btn.classList.add('success')

    // 2秒后恢复原始状态
    setTimeout(() => {
      btn.innerHTML = originalIcon
      btn.classList.remove('success')
    }, 2000)
  })

  console.log('复制成功')
}

// 完全重置分析（包括清空输入框）- 用于重置按钮
function resetAnalysisCompletely() {
  // 清空输入框内容
  document.getElementById("inputContent").value = ""
  document.getElementById("targetContent").value = ""
  
  // 调用常规重置
  resetAnalysis()
}

// 显示通知消息（从generation页面复制）
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

// 自定义图表拖拽调整大小功能
let isResizing = false
let currentResizeTarget = null
let startY = 0
let startHeight = 0
let autoScrollInterval = null

function startChartResize(event, chartContentId) {
  event.preventDefault()
  isResizing = true
  currentResizeTarget = document.getElementById(chartContentId)
  startY = event.clientY
  startHeight = parseInt(window.getComputedStyle(currentResizeTarget).height, 10)
  
  document.addEventListener('mousemove', doChartResize)
  document.addEventListener('mouseup', stopChartResize)
  document.body.style.cursor = 'ns-resize'
  document.body.style.userSelect = 'none'
}

function doChartResize(event) {
  if (!isResizing || !currentResizeTarget) return
  
  const currentY = event.clientY
  const deltaY = currentY - startY
  const newHeight = startHeight + deltaY
  
  // 设置最小高度为200px，最大高度为1000px
  const minHeight = 200
  const maxHeight = 1000
  const clampedHeight = Math.max(minHeight, Math.min(maxHeight, newHeight))
  
  currentResizeTarget.style.height = clampedHeight + 'px'
  
  // 自动滚动页面以跟随拖拽
  autoScrollOnResize(event)
}

function autoScrollOnResize(event) {
  // 清除之前的滚动
  if (autoScrollInterval) {
    clearInterval(autoScrollInterval)
    autoScrollInterval = null
  }
  
  const viewportHeight = window.innerHeight
  const mouseY = event.clientY
  const scrollThreshold = 60 // 距离边缘多少像素时开始滚动
  
  let scrollDirection = 0
  let scrollSpeed = 0
  
  // 如果鼠标接近页面底部，向下滚动
  if (mouseY > viewportHeight - scrollThreshold) {
    scrollDirection = 1
    // 使用更平滑的速度计算
    const distance = mouseY - (viewportHeight - scrollThreshold)
    scrollSpeed = Math.max(1, Math.min(8, distance / 3))
  }
  // 如果鼠标接近页面顶部，向上滚动
  else if (mouseY < scrollThreshold) {
    scrollDirection = -1
    const distance = scrollThreshold - mouseY
    scrollSpeed = Math.max(1, Math.min(8, distance / 3))
  }
  
  // 如果需要滚动，启动更平滑的滚动
  if (scrollDirection !== 0) {
    const smoothScroll = () => {
      if (isResizing && scrollDirection !== 0) {
        window.scrollBy({
          top: scrollDirection * scrollSpeed,
          behavior: 'instant'
        })
        autoScrollInterval = requestAnimationFrame(smoothScroll)
      }
    }
    autoScrollInterval = requestAnimationFrame(smoothScroll)
  }
}

function stopChartResize() {
  isResizing = false
  currentResizeTarget = null
  
  // 清除自动滚动
  if (autoScrollInterval) {
    cancelAnimationFrame(autoScrollInterval)
    autoScrollInterval = null
  }
  
  document.removeEventListener('mousemove', doChartResize)
  document.removeEventListener('mouseup', stopChartResize)
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
}

// 防止页面意外离开时清理事件监听器
window.addEventListener('beforeunload', () => {
  if (isResizing) {
    stopChartResize()
  }
  if (autoScrollInterval) {
    cancelAnimationFrame(autoScrollInterval)
    autoScrollInterval = null
  }
  // 清理图表工具提示
  hideChartTooltip()
})

// 图表工具提示相关函数
let chartTooltip = null

function showChartTooltip(event, tokenText, value, valueType) {
  // 移除现有的工具提示
  hideChartTooltip()
  
  // 创建工具提示元素
  chartTooltip = document.createElement('div')
  chartTooltip.className = 'chart-tooltip'
  chartTooltip.innerHTML = `${valueType}: ${value} (Token: ${tokenText})`
  
  // 设置工具提示样式
  chartTooltip.style.cssText = `
    position: absolute;
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
    z-index: 10000;
    pointer-events: none;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    opacity: 1;
    transform: translateY(-100%);
    margin-top: -8px;
  `
  
  // 获取鼠标位置并定位工具提示
  const rect = event.currentTarget.getBoundingClientRect()
  chartTooltip.style.left = (rect.left + rect.width / 2) + 'px'
  chartTooltip.style.top = (rect.top + window.scrollY) + 'px'
  chartTooltip.style.transform = 'translateX(-50%) translateY(-100%)'
  
  // 添加到页面
  document.body.appendChild(chartTooltip)
}

function hideChartTooltip() {
  if (chartTooltip && chartTooltip.parentNode) {
    chartTooltip.parentNode.removeChild(chartTooltip)
    chartTooltip = null
  }
}
