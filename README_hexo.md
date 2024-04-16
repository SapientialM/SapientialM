# hexo-butterfly 教程
## 命令操作
```bash
// 生成静态网页 本地运行博客
hexo generate && hexo server

// 本地运行博客 端口 5000
hexo s -p 5000

// 运行地址为：localhost:5000

// 写作
hexo new <title>

// 部署
deploy.sh
```

## 目录结构
```
.
├── _config.yml   博客网站配置文件
├── package.json  
├── scaffolds     hexo生成模板
├── source        
|   ├── _drafts
|   └── _posts
└── themes
```

## 文章撰写
### tags与categories
> 参考：https://hexo.io/zh-cn/docs/writing.html

在hexo中使用tags与categories往往需要使用多标签和多分类，这里记录一下它们的用法。
#### tag
用法：
```
tags:
  - 123
  - 456
tags: [123, 456]
```
多标签写法，这2种都是一样的效果，用哪个都可以，建议使用列表[]式，直观清晰。

#### categories
```
# 这是默认的写法，给文章添加一个分类。
categories: 123
# 这会将文章分类123/456子分类目录下。
categories: [123, 456]
这会将文章分类到123/456子分类目录下。
categories:
   - 123
   - 456
多标签写法，文章被分类到123、456以及123的自分类789这3个分类下面，官方指定写法。
categories:
   - [123]
   - [456]
   - [123, 789]
```

