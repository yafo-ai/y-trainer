let compressionRoundCount = 1
let compressionData = null

// 页面加载完成时初始化
document.addEventListener("DOMContentLoaded", () => {
  // 初始化模型选择器
  initModelDropdown()

  console.log("内容压缩页面已加载")
  updateRemoveButtons()

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

  // 构造URL，确保modelName被正确编码
      const url = new URL(API_CONFIG.getUrl('CHANGE_MODEL'))
  url.searchParams.append("model_name", modelName)

  console.log("完整请求URL:", url.toString())

  // 使用新的请求URL
  proxyFetch(url.toString())
    .then((data) => {
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
    })
    .catch((error) => {
      console.error("模型切换请求失败:", error)
      modelNameDisplay.textContent = originalText
      alert("模型切换请求失败，可能是网络问题或API服务不可用")
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

// Add compression round
function addCompressionRound() {
  compressionRoundCount++
  const roundsContainer = document.getElementById("compressionRounds")

  const newRound = document.createElement("div")
  newRound.className = "compression-round"
  newRound.setAttribute("data-round", compressionRoundCount)

  newRound.innerHTML = `
        <div class="round-header">
            <div class="round-title">第 ${compressionRoundCount} 次压缩</div>
            <button class="remove-round-btn" onclick="removeCompressionRound(${compressionRoundCount})">
                <i class="fas fa-times"></i> 移除
            </button>
        </div>
        <div class="params-grid">
            <div class="param-group">
                <label class="param-label">卷积核大小</label>
                <input type="number" class="param-input" name="kernel_size" value="5" min="1" max="10">
            </div>
            <div class="param-group">
                <label class="param-label">Top-K 值</label>
                <input type="number" class="param-input" name="top_k" value="150" min="1" max="200">
            </div>
            <div class="param-group">
                <label class="param-label">分数类型</label>
                <select class="param-input" name="score_type">
                    <option value="1">原始分数</option>
                    <option value="2">分数加权放大</option>
                </select>
            </div>
            <div class="param-group">
                <label class="param-label">最小分数阈值</label>
                <input type="number" class="param-input" name="min_score" value="0" min="0" max="1" step="0.01">
            </div>
        </div>
    `

  roundsContainer.appendChild(newRound)

  // Scroll to the new round
  setTimeout(() => {
    newRound.scrollIntoView({ behavior: "smooth", block: "nearest" })
  }, 10)

  updateRemoveButtons()
}

// Remove compression round
function removeCompressionRound(roundId) {
  const roundElement = document.querySelector(`[data-round="${roundId}"]`)
  if (roundElement) {
    roundElement.remove()
    updateRemoveButtons()
    updateRoundNumbers()
  }
}

// Update remove button visibility
function updateRemoveButtons() {
  const rounds = document.querySelectorAll(".compression-round")
  rounds.forEach((round, index) => {
    const removeBtn = round.querySelector(".remove-round-btn")
    if (rounds.length > 1) {
      removeBtn.style.display = "block"
    } else {
      removeBtn.style.display = "none"
    }
  })
}

// Update round numbers after removal
function updateRoundNumbers() {
  const rounds = document.querySelectorAll(".compression-round")
  rounds.forEach((round, index) => {
    const roundNumber = index + 1
    round.setAttribute("data-round", roundNumber)
    round.querySelector(".round-title").textContent = `第 ${roundNumber} 次压缩`
    const removeBtn = round.querySelector(".remove-round-btn")
    removeBtn.setAttribute("onclick", `removeCompressionRound(${roundNumber})`)
  })
  compressionRoundCount = rounds.length
}

// Get compression parameters
function getCompressionParameters() {
  const rounds = document.querySelectorAll(".compression-round")
  const parameters = []

  rounds.forEach((round, index) => {
    const kernelSize = round.querySelector('input[name="kernel_size"]').value
    const topK = round.querySelector('input[name="top_k"]').value
    const scoreType = round.querySelector('select[name="score_type"]').value
    const minScore = round.querySelector('input[name="min_score"]').value

    parameters.push({
      round: index + 1,
      kernel_size: Number.parseInt(kernelSize),
      top_k: Number.parseInt(topK),
      score_type: Number.parseInt(scoreType),
      min_score: Number.parseFloat(minScore),
    })
  })

  return parameters
}

// Start compression
function startCompression() {
  const inputContent = document.getElementById("inputContent").value.trim()
  const questionContent = document.getElementById("questionContent").value.trim()

  if (!inputContent) {
    showNotification("请先输入需要压缩的内容", "error")
    return
  }

  // Show loading state
  showLoading()

  // Get compression parameters
  const parameters = getCompressionParameters()

  // 使用真实API执行压缩
  performRealCompression(inputContent, questionContent, parameters)
}

// 使用真实API执行文本压缩
function performRealCompression(inputContent, questionContent, parameters) {
  // 将获取到的参数映射到API参数
  const apiRequests = parameters.map((param, index) => {
    // 构建API请求数据
    const requestData = {
      input: inputContent,
      question: questionContent,
      kernel_num: param.kernel_size,
      topk: param.top_k,
      min_scroe: param.min_score, // 注意API中的参数名称是min_scroe而不是min_score
      score_type: param.score_type, // 确保score_type是数字类型
    }

    console.log(`第${index + 1}轮压缩请求数据:`, requestData)

    // 发起API请求
    return fetch(API_CONFIG.getUrl('COMPRESS'), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      mode: "cors",
      body: JSON.stringify(requestData),
    })
      .then((response) => {
        console.log(`第${index + 1}轮压缩响应状态:`, response.status)
        if (!response.ok) {
          throw new Error(`HTTP错误: ${response.status}`)
        }
        return response.json()
      })
      .then((data) => {
        console.log(`第${index + 1}轮压缩响应数据:`, data)
        // 检查API响应
        if (data.code === 200) {
          return {
            roundIndex: index,
            parameters: param,
            apiResponse: data.data,
          }
        } else {
          throw new Error(`API错误: ${data.message}`)
        }
      })
  })

  // 使用Promise.all处理所有请求
  Promise.all(apiRequests)
    .then((results) => {
      // 处理压缩结果
      processCompressionResults(results, inputContent, questionContent)
    })
    .catch((error) => {
      console.error("压缩请求失败:", error)
      showError("压缩请求失败: " + error.message)
    })
}

// 处理压缩API返回的结果
function processCompressionResults(results, inputContent, questionContent) {
  console.log("处理压缩结果:", results)

  // 从第一轮压缩获取token和注意力分数
  const firstResult = results[0]
  const rawAttentionScores = firstResult.apiResponse.attention_scores || []

  // 计算原始分数的最高和最低值
  const validScores = rawAttentionScores.filter((score) => score !== undefined && score !== null)
  const maxScore = validScores.length > 0 ? Math.max(...validScores) : 1
  const minScore = validScores.length > 0 ? Math.min(...validScores) : 0

  console.log(
    `Token注意力分析 - 分数范围: ${minScore.toFixed(4)} 到 ${maxScore.toFixed(4)} (共${validScores.length}个有效分数)`,
  )

  const originalTokens = firstResult.apiResponse.input_tokens.map((token, index) => {
    const originalScore = firstResult.apiResponse.attention_scores[index] || 0

    return {
      id: index,
      text: token,
      attentionScore: originalScore, // 直接使用原始分数
      position: index,
      originalScore: originalScore,
      minScore: minScore,
      maxScore: maxScore,
    }
  })

  // 创建压缩结果数组
  const compressionResults = results.map((result, index) => {
    // 如果不是第一轮，获取前一轮的结果作为原始内容
    const originalText = index === 0 ? inputContent : results[index - 1].apiResponse.compress_text

    return {
      round: index + 1,
      parameters: result.parameters,
      original: originalText,
      compressed: result.apiResponse.compress_text,
      originalLength: originalText.length,
      compressedLength: result.apiResponse.compress_text.length,

      tokenCount: result.apiResponse.input_tokens.length,
      removedTokens:
        (index === 0 ? originalTokens.length : results[index - 1].apiResponse.input_tokens.length) -
        result.apiResponse.input_tokens.length,
    }
  })

  // 显示结果
  displayCompressionResults({
    inputContent,
    questionContent,
    originalTokens,
    compressionResults,
    model: document.getElementById("modelSelect").value,
  })
}

// 显示错误信息
function showError(message) {
  const errorHtml = `
        <div class="empty-state" style="color: #dc2626;">
            <i class="fas fa-exclamation-circle"></i>
            <p>${message}</p>
            <small style="margin-top: 0.5rem; display: block;">请检查网络连接或服务器状态</small>
        </div>
    `

  document.getElementById("originalTokens").innerHTML = errorHtml
  document.getElementById("compressionResultsContainer").innerHTML = errorHtml
}

// Reset compression
function resetCompression() {
  // Confirm reset if there are multiple rounds or input content
  const hasMultipleRounds = document.querySelectorAll(".compression-round").length > 1
  const hasInputContent = document.getElementById("inputContent").value.trim() !== ""

  if ((hasMultipleRounds || hasInputContent) && !confirm("确定要重置所有参数和内容吗？")) {
    return
  }

  document.getElementById("inputContent").value = ""
  document.getElementById("questionContent").value = ""

  // Reset to single round
  const roundsContainer = document.getElementById("compressionRounds")
  roundsContainer.innerHTML = `
        <div class="compression-round" data-round="1">
            <div class="round-header">
                <div class="round-title">第 1 次压缩</div>
                <button class="remove-round-btn" onclick="removeCompressionRound(1)" style="display: none;">
                    <i class="fas fa-times"></i> 移除
                </button>
            </div>
            <div class="params-grid">
                <div class="param-group">
                    <label class="param-label">卷积核大小</label>
                    <input type="number" class="param-input" name="kernel_size" value="5" min="1" max="10">
                </div>
                <div class="param-group">
                    <label class="param-label">Top-K 值</label>
                    <input type="number" class="param-input" name="top_k" value="150" min="1" max="200">
                </div>
                <div class="param-group">
                    <label class="param-label">分数类型</label>
                    <select class="param-input" name="score_type">
                        <option value="1">原始分数</option>
                        <option value="2">分数加权放大</option>
                    </select>
                </div>
                <div class="param-group">
                    <label class="param-label">最小分数阈值</label>
                    <input type="number" class="param-input" name="min_score" value="0" min="0" max="1" step="0.01">
                </div>
            </div>
        </div>
    `

  compressionRoundCount = 1

  // Reset results
  document.getElementById("originalTokens").innerHTML = `
        <div class="empty-state">
            <i class="fas fa-file-text"></i>
            <p>原始内容的Token分析将在这里显示</p>
            <small style="margin-top: 0.5rem; display: block;">输入内容后开始压缩查看分析结果</small>
        </div>
    `

  document.getElementById("compressionResultsContainer").innerHTML = `
        <div class="empty-state">
            <i class="fas fa-compress-alt"></i>
            <p>压缩结果将在这里显示</p>
            <small style="margin-top: 0.5rem; display: block;">配置参数后开始压缩</small>
        </div>
    `

  compressionData = null
}

// Show loading state
function showLoading() {
  const loadingHtml = `
        <div class="loading">
            <div class="spinner"></div>
            <span>正在进行多轮压缩分析，请稍候...</span>
        </div>
    `

  document.getElementById("originalTokens").innerHTML = loadingHtml
  document.getElementById("compressionResultsContainer").innerHTML = loadingHtml
}

// Simple tokenization - 保留此函数以便在需要时使用
function tokenize(text) {
  return text.split(/(\s+|[，。！？；：、])/)
}

// Display compression results
function displayCompressionResults(data) {
  compressionData = data

  // Display original tokens with attention scores
  document.getElementById("originalTokens").innerHTML = renderOriginalTokens(data.originalTokens)

  // Display compression results
  document.getElementById("compressionResultsContainer").innerHTML = renderCompressionResults(data.compressionResults)
}

// Render original tokens with attention highlighting
function renderOriginalTokens(tokens) {
  return tokens
    .map((token) => {
      if (token.text.trim() === "") return token.text

      // 根据真实分数计算百分比（基于分数范围）
      // 使用与attention页面完全相同的颜色生成逻辑
      const backgroundColor = getAttentionColorByRange(token.attentionScore, token.minScore, token.maxScore)
      
      // 计算相对强度用于调试信息
      const scoreRange = token.maxScore - token.minScore
      let scorePercentage = 0
      let relativeIntensity = 0
      if (scoreRange > 0) {
        scorePercentage = ((token.attentionScore - token.minScore) / scoreRange) * 100
        relativeIntensity = scorePercentage / 100
      } else {
        scorePercentage = 50 // 如果所有分数相同，使用50%
        relativeIntensity = 0.5
      }

                      // 固定使用深色文字，不根据背景透明度改变
        const alpha = 0.01 + relativeIntensity * (0.95 - 0.01)
        const textColor = "#1e293b" // 始终使用深色文字

      // 调试信息：显示分数和颜色生成的对应关系
      if (token.id < 5) {
        // 只显示前5个token的调试信息
        console.log(
          `Token "${token.text}": 原始分数=${token.originalScore.toFixed(4)}, 百分比=${scorePercentage.toFixed(1)}%, 强度=${relativeIntensity.toFixed(3)}, 背景色=${backgroundColor}, 文字色=${textColor}`,
        )
      }

      // 为每个token添加动态样式，与attention页面保持一致
      return `<span class="token" 
            data-token-id="${token.id}" 
            data-attention="${token.attentionScore.toFixed(3)}"
            data-original-score="${token.originalScore.toFixed(3)}"
            data-min-score="${token.minScore.toFixed(3)}"
            data-max-score="${token.maxScore.toFixed(3)}"
            data-score-percentage="${scorePercentage.toFixed(1)}"
            data-relative-intensity="${relativeIntensity.toFixed(3)}"
            onclick="highlightToken(${token.id})"
            title="原始分数: ${token.originalScore.toFixed(3)} (${scorePercentage.toFixed(1)}%) | 范围: ${token.minScore.toFixed(3)} - ${token.maxScore.toFixed(3)} | 点击查看详情"
            style="
                position: relative; 
                display: inline-block; 
                white-space: nowrap;
                background-color: ${backgroundColor};
                color: ${textColor};
                border-radius: 0.25rem;
                padding: 0.125rem 0.25rem;
                margin: 3px;
                font-weight: 400;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
            "
        >${token.text}</span>`
    })
    .join("")
}

// Render compression results
function renderCompressionResults(results) {
  return results
    .map(
      (result) => `
        <div class="compression-round-result">
            <div class="round-result-header">
                <div class="round-result-title">第 ${result.round} 轮压缩结果</div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div class="compression-stats">
                        <span>Token数: ${result.tokenCount}</span>
                        <span>移除: ${result.removedTokens}</span>
                    </div>
                    <button class="copy-btn" onclick="copyRoundResult(${result.round}, this)" title="复制第${result.round}轮结果">
                        <i class="fas fa-copy"></i>
                    </button>
                </div>
            </div>
            <div style="margin-bottom: 0.5rem; font-size: 0.75rem; color: #6b7280;">
                参数: 卷积核=${result.parameters.kernel_size}, Top-K=${result.parameters.top_k}, 
                分数类型=${getScoreTypeText(result.parameters.score_type)}, 
                最小分数=${result.parameters.min_score}
            </div>
            <div class="round-result-content">${result.compressed}</div>
            </div>
    `,
    )
    .join("")
}

// 复制指定轮次的压缩结果
function copyRoundResult(roundNumber, buttonElement) {
  if (!compressionData || !compressionData.compressionResults) {
    console.error("没有压缩结果数据")
    return
  }

  // 查找指定轮次的结果
  const roundResult = compressionData.compressionResults.find((result) => result.round === roundNumber)
  if (!roundResult) {
    console.error("找不到第", roundNumber, "轮的压缩结果")
    return
  }

  // 复制该轮的压缩内容
  copyTextWithFeedback(roundResult.compressed, buttonElement)
}

// 复制功能 - 从generation页面复制
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

// Get score type text
function getScoreTypeText(scoreType) {
  const types = {
    1: "原始分数",
    2: "分数加权放大",
    1: "原始分数",
    2: "分数加权放大",
  }
  return types[scoreType] || `未知(${scoreType})`
}

// 根据动态范围生成注意力颜色（从attention页面复制）
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

// 与attention页面统一的颜色生成函数
function getAttentionColorByIntensity(intensity, originalScore) {
  // 确保强度在0-1之间
  intensity = Math.max(0, Math.min(1, intensity))

  // 使用青色，与attention页面一致
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

// 统一的清除token注意力颜色函数（从attention页面复制）
function clearTokenAttentionColors(containerId) {
  document.querySelectorAll(`#${containerId} .token[data-token-id]`).forEach((token) => {
    token.className = "token"
    token.title = ""
    token.style.backgroundColor = ""
    token.style.color = ""
  })
}

// 显示token注意力分数，不改变高亮或透明度
function highlightToken(tokenId) {
  if (!compressionData) return

  // 立即移除所有现有的分数显示
  const existingDisplays = document.querySelectorAll(".token-score-display-container")
  for (let i = 0; i < existingDisplays.length; i++) {
    const el = existingDisplays[i]
    if (el.parentNode) {
      el.parentNode.removeChild(el)
    }
  }

  // 获取选中的token
  const selectedToken = document.querySelector(`[data-token-id="${tokenId}"]`)
  if (selectedToken) {
    // 获取token数据
    const tokenData = compressionData.originalTokens.find((t) => t.id === tokenId)
    if (tokenData) {
      // 立即显示分数，无延迟
      showSimpleScoreDisplay(selectedToken, tokenData)
    }
  }
}

// Show simple score display next to the token
function showSimpleScoreDisplay(tokenElement, tokenData) {
  // 根据真实分数计算百分比（基于分数范围）
  const scoreRange = tokenData.maxScore - tokenData.minScore
  let scorePercentage = 0
  if (scoreRange > 0) {
    scorePercentage = ((tokenData.attentionScore - tokenData.minScore) / scoreRange) * 100
  } else {
    scorePercentage = 50 // 如果所有分数相同，使用50%
  }

  // 根据真实分数百分比确定注意力等级和颜色（与attention页面统一的青色系统）
  let attentionLevel = "极低"
  let attentionColor = "#02a9a9" // 统一使用青色 #02a9a9，与attention页面保持一致
  if (scorePercentage >= 80) {
    attentionLevel = "极高"
    attentionColor = "#02a9a9" // 青色
  } else if (scorePercentage >= 60) {
    attentionLevel = "高"
    attentionColor = "#02a9a9" // 青色
  } else if (scorePercentage >= 40) {
    attentionLevel = "中等"
    attentionColor = "#02a9a9" // 青色
  } else if (scorePercentage >= 20) {
    attentionLevel = "低"
    attentionColor = "#02a9a9" // 青色
  }

  // 获取token的位置信息
  const tokenRect = tokenElement.getBoundingClientRect()

  // 创建分数显示容器
  const container = document.createElement("div")
  container.className = "token-score-display-container"
  container.setAttribute("data-token-id", tokenData.id)

  // 设置容器样式 - 使用fixed定位相对于视口
  container.style.cssText = `
        position: fixed !important;
        z-index: 2147483647 !important;
        pointer-events: none !important;
        left: ${tokenRect.left + tokenRect.width / 2}px !important;
        top: ${tokenRect.bottom}px !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 1 !important;
        visibility: visible !important;
    `

  // 创建分数显示元素
  const scoreDisplay = document.createElement("div")
  scoreDisplay.className = "token-score-display"

  // 设置分数显示的内容
  const contentDiv = document.createElement("div")
  contentDiv.style.cssText = "display: flex; align-items: center; gap: 0.5rem;"

  const scoreText = document.createElement("span")
  // 显示原始分数、百分比和分数范围
  scoreText.textContent = `分数: ${tokenData.originalScore.toFixed(4)} (${scorePercentage.toFixed(1)}%) | 范围: ${tokenData.minScore.toFixed(4)} - ${tokenData.maxScore.toFixed(4)}`
  scoreText.style.cssText = "color: #374151; font-weight: 500; font-size: 0.75rem;"

  const levelBadge = document.createElement("span")
  levelBadge.textContent = attentionLevel
  levelBadge.style.cssText = `
        background: ${attentionColor}; 
        color: white; 
        padding: 0.125rem 0.375rem; 
        border-radius: 1rem; 
        font-size: 0.7rem; 
        font-weight: 600;
    `

  // 添加指向token的箭头
  const arrow = document.createElement("div")
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

  // 将容器添加到body而不是token元素
  document.body.appendChild(container)

  // 3秒后自动移除
  const autoRemoveTimer = setTimeout(() => {
    if (container.parentNode) {
      container.parentNode.removeChild(container)
    }
  }, 3000)

  // 点击其他位置时移除分数显示
  const removeScoreDisplay = (event) => {
    if (event.target !== tokenElement && !tokenElement.contains(event.target)) {
      if (container.parentNode) {
        container.parentNode.removeChild(container)
      }
      document.removeEventListener("click", removeScoreDisplay)
      clearTimeout(autoRemoveTimer)
    }
  }
  document.addEventListener("click", removeScoreDisplay)

  // 按ESC键移除分数显示
  const handleEscape = (event) => {
    if (event.key === "Escape") {
      if (container.parentNode) {
        container.parentNode.removeChild(container)
      }
      document.removeEventListener("keydown", handleEscape)
      clearTimeout(autoRemoveTimer)
    }
  }
  document.addEventListener("keydown", handleEscape)

  // 添加滚动事件监听器，确保分数显示跟随token
  const updatePosition = () => {
    const updatedRect = tokenElement.getBoundingClientRect()
    container.style.left = `${updatedRect.left + updatedRect.width / 2}px`
    container.style.top = `${updatedRect.bottom}px`
  }
  window.addEventListener("scroll", updatePosition)
  window.addEventListener("resize", updatePosition)

  // 在分数显示被移除时清除事件监听器
  const originalRemoveChild = container.parentNode ? container.parentNode.removeChild : null
  if (originalRemoveChild) {
    container.parentNode.removeChild = function (element) {
      if (element === container) {
        window.removeEventListener("scroll", updatePosition)
        window.removeEventListener("resize", updatePosition)
      }
      return originalRemoveChild.call(this, element)
    }
  }
}

// Toggle parameters visibility
function toggleParamsVisibility(event) {
  event.stopPropagation()
  const container = document.getElementById("compressionParamsContainer")
  const icon = document.getElementById("toggleParamsIcon")

  container.classList.toggle("collapsed")

  if (container.classList.contains("collapsed")) {
    icon.className = "fas fa-chevron-down"
  } else {
    icon.className = "fas fa-chevron-up"
  }
}

// Load input sample
function loadInputSample() {
  const inputContent = document.getElementById("inputContent")
  inputContent.value = `人工智能（Artificial Intelligence，简称AI）是一门研究如何使计算机模拟人类智能的学科。它包括多个分支，如机器学习、深度学习、自然语言处理、计算机视觉等。

机器学习是AI的核心技术，它使计算机能够从数据中学习并提高性能，而无需明确编程。深度学习是机器学习的一个子集，它使用神经网络来模拟人脑的工作方式。

当前，AI技术已广泛应用于各个领域，如医疗诊断、金融分析、自动驾驶、智能客服等。尽管AI取得了显著进步，但它仍面临诸多挑战，如数据隐私、算法偏见、安全性等问题。

未来，随着技术的不断发展，AI有望在更多领域发挥重要作用，为人类社会带来更多便利和创新。`
}

// Load question sample
function loadQuestionSample() {
  const questionContent = document.getElementById("questionContent")
  questionContent.value = `人工智能`
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
