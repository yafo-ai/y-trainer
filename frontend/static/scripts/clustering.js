const currentAlgorithm = "kmeans"
let clusteringResults = null

// 输入验证相关函数
function validateArrayInput(input) {
  const result = {
    isValid: false,
    data: null,
    error: null
  }
  
  // 检查是否为空
  if (!input || !input.trim()) {
    result.error = "输入不能为空"
    return result
  }
  
  try {
    // 尝试解析JSON
    const parsed = JSON.parse(input.trim())
    
    // 检查是否为数组
    if (!Array.isArray(parsed)) {
      result.error = "输入必须是数组格式，例如：[\"文本1\", \"文本2\"]"
      return result
    }
    
    // 检查数组是否为空
    if (parsed.length === 0) {
      result.error = "数组不能为空，至少需要一个元素"
      return result
    }
    
    // 检查数组元素是否都是字符串
    for (let i = 0; i < parsed.length; i++) {
      if (typeof parsed[i] !== 'string') {
        result.error = `数组第${i + 1}个元素必须是字符串，当前是${typeof parsed[i]}类型`
        return result
      }
      
      // 检查字符串是否为空
      if (!parsed[i].trim()) {
        result.error = `数组第${i + 1}个元素不能为空字符串`
        return result
      }
    }
    
    result.isValid = true
    result.data = parsed
    return result
    
  } catch (error) {
    // JSON解析错误
    if (error instanceof SyntaxError) {
      result.error = "JSON格式错误：" + getJsonErrorMessage(error.message, input)
    } else {
      result.error = "输入格式无效：" + error.message
    }
    return result
  }
}

// 获取更友好的JSON错误消息
function getJsonErrorMessage(errorMessage, input) {
  if (errorMessage.includes("Unexpected token")) {
    if (!input.trim().startsWith('[')) {
      return "必须以 [ 开头"
    }
    if (!input.trim().endsWith(']')) {
      return "必须以 ] 结尾"
    }
    if (errorMessage.includes("Unexpected token '")) {
      return "存在语法错误，请检查引号、逗号是否正确"
    }
  }
  if (errorMessage.includes("Unexpected end")) {
    return "JSON不完整，可能缺少结束符号"
  }
  return "请检查JSON语法是否正确"
}

// 显示验证消息
function showValidationMessage(message, type = 'error') {
  const validationDiv = document.getElementById('validationMessage')
  const textarea = document.getElementById('dataInput')
  
  if (message) {
    validationDiv.className = `validation-message ${type}`
    validationDiv.innerHTML = `<i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'check-circle'}"></i>${message}`
    textarea.className = `data-textarea ${type}`
  } else {
    validationDiv.className = 'validation-message'
    validationDiv.innerHTML = ''
    textarea.className = 'data-textarea'
  }
}

// 实时验证输入
function setupInputValidation() {
  const textarea = document.getElementById('dataInput')
  let validationTimeout = null
  
  textarea.addEventListener('input', function() {
    // 清除之前的超时
    if (validationTimeout) {
      clearTimeout(validationTimeout)
    }
    
    // 延迟验证，避免频繁触发
    validationTimeout = setTimeout(() => {
      const input = this.value.trim()
      
      if (!input) {
        showValidationMessage(null)
        return
      }
      
      const validation = validateArrayInput(input)
      if (validation.isValid) {
        showValidationMessage(`有效的JSON数组，包含${validation.data.length}个元素`, 'success')
      } else {
        showValidationMessage(validation.error, 'error')
      }
    }, 500) // 500ms延迟
  })
  
  // 失去焦点时也进行验证
  textarea.addEventListener('blur', function() {
    const input = this.value.trim()
    if (input) {
      const validation = validateArrayInput(input)
      if (validation.isValid) {
        showValidationMessage(`有效的JSON数组，包含${validation.data.length}个元素`, 'success')
      } else {
        showValidationMessage(validation.error, 'error')
      }
    }
  })
}

// Load example data
function loadExampleData() {
  const exampleData = [
    "用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：连续能打印多长时间",
    "用户正在浏览：爱普生L3256三合一彩色喷墨机（A4）[产品id:5A2655]用户：3253呢",
    "用户正在浏览：爱普生L11058墨仓式彩色喷墨打印机（A3）[产品id:5A2679]用户：这个能自动打双面吗？",
    "用户正在浏览：爱普生L11058墨仓式彩色喷墨打印机（A3）[产品id:5A2679]用户：自动双面吗？",
    "用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：连续打印？",
    "用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：用啥耗材",
    "用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：使用什么耗材",
    "用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：适用什么耗材？",
    "用户正在浏览：爱普生L3256三合一彩色喷墨打印机（A4）[产品id:5A2655]用户：耗材型号",
  ]

  const jsonString = JSON.stringify(exampleData, null, 2)
  document.getElementById("dataInput").value = jsonString
  
  // 触发验证显示
  const validation = validateArrayInput(jsonString)
  if (validation.isValid) {
    showValidationMessage(`有效的JSON数组，包含${validation.data.length}个元素`, 'success')
  } else {
    showValidationMessage(validation.error, 'error')
  }
}

// Start clustering
function startClustering() {
  const dataInput = document.getElementById("dataInput").value.trim()

  if (!dataInput) {
    showValidationMessage("请先输入数据", "error")
    return
  }

  // 验证输入格式
  const validation = validateArrayInput(dataInput)
  if (!validation.isValid) {
    showValidationMessage(validation.error, "error")
    return
  }

  // Show loading state
  showLoading()

  // Get parameters
  const filterScore = Number.parseFloat(document.getElementById("filterScore").value)
  const useAttention = document.getElementById("attentionExtraction").checked

  // 使用验证后的数据执行聚类
  performRealClustering(validation.data, filterScore, useAttention)
}

// 使用真实API执行文本聚类
function performRealClustering(dataArray, filterScore, useAttention) {
  // 现在传入的是已经验证的数组数据
  const lines = dataArray
  console.log("使用验证后的数组数据，包含", lines.length, "条记录")

  // 准备API请求数据
  const requestData = {
    data: lines,
    is_attent: useAttention,
    score_threshold: filterScore || 0,
  }

  console.log("发送聚类请求数据:", requestData)

  // 发送请求到聚类API
      fetch(API_CONFIG.getUrlWithQuery('CLUSTERS', {default: 1500}), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    mode: "cors",
    body: JSON.stringify(requestData),
  })
    .then((response) => {
      console.log("聚类API响应状态:", response.status)
      if (!response.ok) {
        throw new Error(`HTTP错误: ${response.status}`)
      }
      return response.json()
    })
    .then((data) => {
      console.log("聚类API响应数据:", data)
      if (data.code === 200) {
        // 处理聚类结果
        processClusteringResults(data.data, lines, filterScore, useAttention)
      } else {
        console.error("聚类API返回错误:", data.message)
        showError("聚类API返回错误: " + data.message)
      }
    })
    .catch((error) => {
      console.error("聚类请求失败:", error)
      showError("聚类请求失败: " + error.message)

      // 如果API请求失败，回退到模拟数据
      console.log("回退到模拟聚类")
      const results = performMockClustering(lines, filterScore, useAttention)
      displayResults(results)
    })
}

// 处理聚类API返回的结果
function processClusteringResults(apiResult, inputLines, filterScore, useAttention) {
  // 创建自定义的聚类对象
  const clusterIds = Object.keys(apiResult.cluster_dict)
  const clusters = clusterIds
    .map((clusterId, index) => {
      const clusterItems = apiResult.cluster_dict[clusterId]
      
      // 为-1聚类使用特殊名称
      const clusterName = clusterId === "-1" ? "未分类" : `聚类 ${index}`;

      return {
        id: index,
        name: clusterName,
        originalId: clusterId,
        items: clusterItems.map((content, itemIndex) => {
          return {
            id: `${clusterId}-${itemIndex}`,
            content: content,
          }
        }),
        color: getClusterColor(index),
      }
    })

  // 显示结果
  displayResults({
    result: apiResult,
    clusters: clusters,
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

  document.getElementById("resultsContent").innerHTML = errorHtml
}

// Show loading state
function showLoading() {
  const loadingHtml = `
        <div class="loading">
            <div class="spinner"></div>
            <span>正在进行聚类分析，请稍候...</span>
        </div>
    `

  document.getElementById("resultsContent").innerHTML = loadingHtml
}

// Perform mock clustering (保留为备用方案)
function performMockClustering(dataArray, filterScore, useAttention) {
  // 现在传入的是已经验证的数组数据
  const lines = dataArray

  // 创建聚类（固定3个聚类）
  const clusterCount = 3
  const clusters = Array.from({ length: clusterCount }, (_, i) => ({
    id: i,
    name: `聚类 ${i}`,
    items: [],
    center: null,
    color: getClusterColor(i),
  }))

  // 为每条数据分配聚类
  const clusterAssignments = []
  const includedItems = []

  lines.forEach((line, index) => {
    if (typeof line !== "string") {
      line = String(line)
    }

    const clusterIndex = Math.floor(Math.random() * clusterCount)
    const similarity = useAttention
      ? Math.random() * 0.3 + 0.7
      : // 0.7-1.0 for attention extraction
        Math.random() * 0.5 + 0.5 // 0.5-1.0 otherwise

    clusterAssignments.push(clusterIndex)

    // 只添加相似度高于过滤分数的项目
    if (similarity >= filterScore) {
      clusters[clusterIndex].items.push({
        id: index,
        content: line.trim(),
      })
      includedItems.push(index)
    }
  })

  // 生成随机样本
  const randomSampleCount = Math.min(3, lines.length)
  const randomSamples = []
  for (let i = 0; i < randomSampleCount; i++) {
    const randomIndex = Math.floor(Math.random() * lines.length)
    randomSamples.push(lines[randomIndex])
  }

  // 创建cluster_dict对象
  const cluster_dict = {}
  clusters.forEach((cluster, index) => {
    cluster_dict[index] = cluster.items.map((item) => item.content)
  })

  // 构建聚类结果对象
  const result = {
    cluster_dict: cluster_dict,
    clusters: clusterAssignments,
    data: lines,
    random_samples: randomSamples,
  }

  return { result, clusters }
}

// Get cluster color
function getClusterColor(index) {
  const colors = ["#6366f1", "#8b5cf6", "#ec4899", "#ef4444", "#f59e0b", "#10b981", "#06b6d4", "#84cc16"]
  return colors[index % colors.length]
}

// Display results
function displayResults(results) {
  clusteringResults = results
  const { result, clusters } = results

  // 显示主要结果
  const validClusters = clusters.filter((cluster) => cluster.items.length > 0)
  const resultsContent = document.getElementById("resultsContent")

  if (validClusters.length === 0) {
    resultsContent.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>没有找到聚类结果</p>
                <small>请尝试调整参数或检查输入数据</small>
            </div>
        `
    return
  }

  // 创建结果展示的HTML
  let html = ""

  // 添加聚类结果可视化
  html += `<div class="result-section">
        <h3 style="margin-bottom: 1rem; font-size: 1rem; color: #374151;">聚类结果可视化</h3>
        ${validClusters
          .map(
            (cluster) => `
            <div class="cluster-group">
                <div class="cluster-header">
                    <div class="cluster-title">
                        <div style="width: 12px; height: 12px; background: ${cluster.color}; border-radius: 2px;"></div>
                        聚类 ${cluster.id + 1}
                    </div>
                    <div class="cluster-badge">${cluster.items.length} 项</div>
                </div>
                <div class="cluster-items">
                    ${cluster.items
                      .map(
                        (item) => `
                        <div class="cluster-item">
                            <div>${item.content}</div>
                        </div>
                    `,
                      )
                      .join("")}
                </div>
            </div>
        `,
          )
          .join("")}
    </div>`



  // 添加JSON格式输出 - 保留完整数据
  html += `<div class="result-section" style="margin-top: 2rem;">
        <h3 style="margin-bottom: 1rem; font-size: 1rem; color: #374151;">JSON格式输出</h3>
        <div style="position: relative;">
            <button onclick="copyJsonResults(this)" class="input-btn" style="position: absolute; top: 0.5rem; right: 0.5rem;padding-right: 15px;">
                <i class="fas fa-copy"></i> 复制
            </button>
            <pre style="background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; overflow: auto; font-family: monospace; font-size: 0.8rem;">${JSON.stringify(result, null, 2)}</pre>
        </div>
    </div>`

  resultsContent.innerHTML = html
}

// 复制JSON结果到剪贴板
function copyJsonResults(buttonElement) {
  if (clusteringResults && clusteringResults.result) {
    // 使用原始完整JSON结果，确保包含-1聚类数据
    const jsonString = JSON.stringify(clusteringResults.result, null, 2)
    copyTextWithFeedback(jsonString, buttonElement)
  }
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

// Clear data
function clearData() {
  document.getElementById("dataInput").value = ""
  document.getElementById("filterScore").value = "1500"
  document.getElementById("attentionExtraction").checked = true

  // 清除验证提示消息
  showValidationMessage(null)

  clusteringResults = null

  document.getElementById("resultsContent").innerHTML = `
        <div class="empty-state">
            <i class="fas fa-project-diagram"></i>
            <p>请输入数据并点击"聚类"按钮</p>
        </div>
    `
}

// 页面加载完成时初始化
document.addEventListener("DOMContentLoaded", () => {
  // 初始化模型选择器
  initModelDropdown()

  // 设置输入验证
  setupInputValidation()

  console.log("聚类筛选页面已加载")
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
