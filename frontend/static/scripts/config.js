// API配置文件
// 用于集中管理所有接口的IP地址和端口号

// 手动配置API服务器地址 - 如果API服务器和前端不在同一地址，请取消注释并设置
window.MANUAL_API_URL = '';

const API_CONFIG = {
    // 基础API地址 - 服务器环境
    BASE_URL: (() => {
        // 如果手动设置了API地址，直接使用
        if (window.MANUAL_API_URL) {
            console.log('使用手动配置的API地址:', window.MANUAL_API_URL);
            return window.MANUAL_API_URL;
        }
        
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        const host = window.location.host;
        
        console.log('环境检测:', { hostname, protocol, host, href: window.location.href });
        
        // 服务器环境，使用当前域名
        const serverUrl = protocol + '//' + host;
        console.log('使用服务器环境地址:', serverUrl);
        return serverUrl;
    })(),
    
    ENDPOINTS: {
        CURRENT_MODEL_NAME: '/api/base/current_model_name',
        GET_MODELS: '/api/base/get_models',
        CHANGE_MODEL: '/api/base/change_model',
        ATTENTION_SCORES: '/api/feature/attention_scores',
        LOSS_SCORES: '/api/feature/loss_scores',
        ENTROPY_SCORES: '/api/feature/entropy_scores',
        COMPRESS: '/api/feature/compress',
        GENERATE_SAMPLES: '/api/feature/generate_samples',
        DYNAMIC_GENERATE_SAMPLES: '/api/feature/dynamic_generate_samples',
        CLUSTERS: '/api/feature/clusters'
    },
    
    // 构建完整URL的辅助方法
    getUrl: function(endpoint, params = '') {
        const url = this.BASE_URL + this.ENDPOINTS[endpoint];
        const fullUrl = params ? `${url}${params}` : url;
        console.log(`构建URL: ${endpoint} -> ${fullUrl}`);
        return fullUrl;
    },
    
    // 构建完整URL（带查询参数）
    getUrlWithQuery: function(endpoint, queryParams = {}) {
        const baseUrl = this.BASE_URL + this.ENDPOINTS[endpoint];
        const queryString = Object.keys(queryParams).length > 0 
            ? '?' + Object.entries(queryParams).map(([key, value]) => `${key}=${value}`).join('&')
            : '';
        return baseUrl + queryString;
    }
};

// 兼容性：为了向后兼容，也可以直接导出配置对象
window.API_CONFIG = API_CONFIG; 



function $ajaxGet(url, data, sucfn, errfn) {
    var queryParams = data;
    var queryString = new URLSearchParams(queryParams).toString();
    var ajaxurl = url + (data ? '?' + queryString : '');
    fetch(ajaxurl)
        .then(async response => {
            if (response.body && response.body.getReader) {
                const reader = response.body.getReader();
                let result = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        break;
                    }
                    result += new TextDecoder().decode(value);
                }
                var res = JSON.parse(result);
                if (res.code == 200) {
                    return res;
                } else {
                    throw new Error(result);
                }

            } else if (response.status == 200) {
                return response.json()
            }

        })
        .then(data => {

            sucfn && sucfn(data)
        })
        .catch(error => {
            console.log(error, 'error');
            try {
                var errdata = JSON.parse(error.toString().replace(/^Error: /, ''));
                if (errdata.code != 200 && errdata.message) {
                    alert(errdata.message)
                }
            } catch (err) {
                console.log(err)
            }

            errfn && errfn(error)
        });
}

function $ajax(url, data, sucfn, errfn) {
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
        .then(async response => {
            if (response.body && response.body.getReader) {
                const reader = response.body.getReader();
                let result = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        break;
                    }
                    result += new TextDecoder().decode(value);
                }
                var res = JSON.parse(result);
                if (res.code == 200) {
                    return res;
                } else {
                    throw new Error(result);
                }
            } else if (response.status == 200) {
                return response.json()
            }
        })
        .then(data => {
            sucfn && sucfn(data)
        })
        .catch(error => {
            try {
                var errdata = JSON.parse(error.toString().replace(/^Error: /, ''));
                if (errdata.code != 200 && errdata.message) {
                    alert(errdata.message)
                }
            } catch (err) {
                console.log(err)
            }
            errfn && errfn(error)
        });

}
