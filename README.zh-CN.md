# Hugging Face WebDAV Gateway 中文说明

将一个或多个 Hugging Face 仓库以只读 WebDAV 的形式统一暴露出来。

每个已配置仓库会映射为 WebDAV 根目录下的一个顶级目录，例如：

- `/dav/models` -> `owner/model-repo`
- `/dav/datasets` -> `owner/dataset-repo`
- `/dav/spaces-assets` -> `owner/space-repo`

适合用 Windows 资源管理器、macOS Finder、rclone、davfs2、OpenList 等 WebDAV 客户端访问 Hugging Face 上的模型、数据集或 Space 文件。

## 功能特性

- 只读 WebDAV 视图
- 支持同时挂载多个 Hugging Face 仓库
- 支持 `model`、`dataset`、`space` 三种仓库类型
- 使用本地 Hugging Face 缓存保存下载文件
- 支持 YAML 或环境变量配置
- 默认启用 Basic Auth，账号密码为 `admin / admin`
- 首页 `/` 提供挂载信息展示，WebDAV 实际入口为 `/dav`

## 项目结构

- `config.yaml` - Docker / Spaces 使用的默认配置文件
- `requirements.txt` - Python 依赖
- `run.py` - 启动入口
- `Dockerfile` - Docker 与 Hugging Face Spaces 容器镜像
- `docker-compose.yml` - 本地 Docker Compose 示例
- `src/hf_webdav_gateway/config.py` - 配置加载与校验
- `src/hf_webdav_gateway/provider.py` - WebDAV Provider 实现
- `src/hf_webdav_gateway/server.py` - WSGI 服务与认证逻辑

## 快速开始

1. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. 通过 `config.yaml` 或环境变量配置仓库。

编辑 `config.yaml`：

```bash
notepad config.yaml
```

或者直接设置环境变量：

```bash
set HF_WEBDAV_HOST=127.0.0.1
set HF_WEBDAV_PORT=8080
set HF_WEBDAV_REPOSITORIES=models|openai/whisper-large-v3|model|main;datasets|HuggingFaceFW/fineweb|dataset|main
```

如果访问私有仓库，还可以设置 token：

```bash
set HF_TOKEN=hf_xxx
```

3. 启动服务：

```bash
python run.py --config config.yaml
```

4. 浏览器访问首页：

```text
http://127.0.0.1:8080/
```

5. WebDAV 客户端连接地址：

```text
http://127.0.0.1:8080/dav
```

## HF_WEBDAV_REPOSITORIES 参数说明

格式如下：

```text
alias|repo_id|repo_type|revision[|token_env];alias2|repo_id2|repo_type2|revision2[|token_env2]
```

说明：

- `alias` - WebDAV 中显示的目录名，例如 `models`
- `repo_id` - Hugging Face 仓库 ID，格式通常为 `owner/name`
- `repo_type` - 仓库类型，可选 `model`、`dataset`、`space`，也兼容常见复数写法 `models`、`datasets`、`spaces`
- `revision` - 分支、tag 或 commit，通常写 `main`
- `token_env` - 可选，表示从哪个环境变量读取访问 token，例如 `HF_TOKEN`

示例：

```text
models|openai/whisper-large-v3|model|main;datasets|HuggingFaceFW/fineweb|dataset|main;private-models|your-org/secret-model|model|main|HF_TOKEN
```

这会挂载出：

- `/dav/models`
- `/dav/datasets`
- `/dav/private-models`

## 认证说明

`/dav` 默认启用 HTTP Basic Auth。

默认账号密码：

```text
admin / admin
```

可通过环境变量覆盖：

```bash
set HF_WEBDAV_USERNAME=admin
set HF_WEBDAV_PASSWORD=change-me
```

行为说明：

- `/dav` 始终需要账号密码
- `/` 首页公开可访问
- `/healthz` 健康检查公开可访问
- 已显式关闭 WsgiDAV 内置认证，仅使用外层自定义认证

## Docker 运行

构建并运行：

```bash
docker build -t hf-webdav-gateway .
docker run --rm -p 8080:7860 -e HF_WEBDAV_USERNAME=admin -e HF_WEBDAV_PASSWORD=admin -e HF_WEBDAV_REPOSITORIES="models|openai/whisper-large-v3|model|main;datasets|HuggingFaceFW/fineweb|dataset|main" -v %cd%/.hf_cache:/data/hf-home hf-webdav-gateway
```

或使用 Compose：

```bash
docker compose up --build
```

说明：

- 容器默认监听 `7860`
- Hugging Face 缓存目录为 `/data/hf-home`
- 浏览器查看首页用 `/`
- WebDAV 客户端连接 `/dav`

## Hugging Face Spaces 部署

本项目可直接作为 Docker Space 运行。

建议：

- 保持仓库根目录存在 `Dockerfile`
- 保持仓库根目录存在 `README.md`
- 在 Space 的 `Variables` 或 `Secrets` 中配置 `HF_WEBDAV_REPOSITORIES`
- 私有仓库映射建议放到 `Secrets`
- 如果同名变量同时存在于 `Variables` 和 `Secrets`，通常可认为 `Secrets` 优先

推荐配置示例：

```text
HF_WEBDAV_REPOSITORIES=models|openai/whisper-large-v3|model|main;spaces-assets|username/my-space-assets|space|main
HF_WEBDAV_USERNAME=admin
HF_WEBDAV_PASSWORD=admin
```

## 注意事项

- 当前实现为只读，不支持上传回 Hugging Face
- 文件内容通过 `huggingface_hub` 拉取并缓存
- 若没有配置任何仓库，服务仍可启动，但首页会显示空挂载列表
- 若用于生产环境，建议前面加 Nginx / Caddy 等反向代理

## 后续可扩展方向

- 更细粒度的访问控制
- 写入支持，映射到 Hugging Face commit API
- 元数据缓存 TTL 配置
- 更丰富的首页状态展示
