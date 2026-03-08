# Hugging Face WebDAV Gateway 中文说明

[English](README.md)

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

2. 通过 `config.yaml` 配置服务端参数，再通过 `HF_WEBDAV_REPOSITORIES` 发现一个或多个用户的仓库。

编辑 `config.yaml`：

```bash
notepad config.yaml
```

然后填写一个或多个用户条目：

```bash
set HF_WEBDAV_REPOSITORIES=smanx|hf_xxx;other-user|hf_yyy
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

## 路径结构

自动发现后的固定挂载路径如下：

```text
/dav/<用户名>/models/<仓库名>
/dav/<用户名>/datasets/<仓库名>
/dav/<用户名>/spaces/<仓库名>
```

例如：

```text
/dav/smanx/models/my-model
/dav/smanx/datasets/phoenix-data
/dav/smanx/spaces/demo-space
```

## HF_WEBDAV_REPOSITORIES 参数说明

现在只保留用户发现模式：

```text
HF_WEBDAV_REPOSITORIES=hf_xxx;hf_yyy
```

多个条目之间支持以下分隔符：

- `;`
- `,`
- 换行

逗号示例：

```text
HF_WEBDAV_REPOSITORIES=hf_xxx,hf_yyy
```

换行示例：

```text
HF_WEBDAV_REPOSITORIES=hf_xxx
hf_yyy
```

程序会优先把 `HF_WEBDAV_REPOSITORIES` 中的每个条目当作 Hugging Face token 处理。如果 token 能成功识别，就自动反查用户名；如果 token 不存在或无效，就把该条目当作用户名处理，并仅查询公开仓库。

- 不需要手动指定 `repo_type`
- 不需要手动指定路径
- 不需要手动指定仓库名列表
- 路径固定为 `/用户名/models/仓库名`、`/用户名/datasets/仓库名`、`/用户名/spaces/仓库名`
- 支持多个用户名，条目之间可使用 `;`、`,` 或换行分隔
- 推荐直接写 `token`
- 如果某个条目不是有效 token，就会按用户名处理
- 兼容 `用户名|token` 这种写法

如果不想启用自动发现，可以不设置 `HF_WEBDAV_REPOSITORIES`，或者设为以下任意值：

```text
0
false
no
off
disable
disabled
```

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
docker run --rm -p 8080:7860 -e HF_WEBDAV_USERNAME=admin -e HF_WEBDAV_PASSWORD=admin -e HF_WEBDAV_REPOSITORIES="hf_xxx;hf_yyy" -v %cd%/.hf_cache:/data/hf-home hf-webdav-gateway
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
HF_WEBDAV_REPOSITORIES=hf_xxx;hf_yyy
HF_WEBDAV_USERNAME=admin
HF_WEBDAV_PASSWORD=admin
```

## 注意事项

- 当前实现为只读，不支持上传回 Hugging Face
- 文件内容通过 `huggingface_hub` 拉取并缓存
- 若启用了 `HF_WEBDAV_REPOSITORIES`，程序会自动发现这些用户的所有可见仓库
- 若用于生产环境，建议前面加 Nginx / Caddy 等反向代理

## 后续可扩展方向

- 更细粒度的访问控制
- 写入支持，映射到 Hugging Face commit API
- 元数据缓存 TTL 配置
- 更丰富的首页状态展示
