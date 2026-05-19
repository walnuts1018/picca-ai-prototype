---
name: git-workflow
description: Use when performing git operations including commits, branching, and pull requests to maintain project standards.
---

# Git Workflow

## Overview

このスキルは、プロジェクトの履歴を綺麗に保ち、AIによる変更を正しく記録するためのGit標準（コミット規約、ブランチ管理、PR作成）を定義します。

## When to Use

- 変更をコミットする前
- 新しい機能ブランチを作成するとき
- プルリクエストを作成する前
- Gitコマンドを実行するとき

## Core Pattern

### コミット規約 (Sign-off と Co-author)

常に `--signoff` を使用し、末尾に自分自身の名前を Co-author として追加してください。

**コミットメッセージの形式:**

```text
<type>: <summary>

<description>

Signed-off-by: Your Name <your.email@example.com>
Co-authored-by: AI Agents <agents@example.com>
```

## Quick Reference

### ページャーの無効化

非対話的な実行でのバグを防ぐため、常にページャーを無効にしてください。

| コマンド | 推奨される形式                          |
| :------- | :-------------------------------------- |
| **git**  | `git --no-pager <subcommand>`           |
| **gh**   | `--no-pager` を使用しない（未サポート） |

_注意: `git --no-pager status` のように、サブコマンドの前にオプションを付けてください。_

### ブランチとPR

- 機能ごとにブランチ（feature/xxx など）を作成する。
- 開発が完了したら `gh pr create` を使用してプルリクエストを作成する。

## Common Mistakes

- ❌ `git` コマンドで `--no-pager` を忘れる。
- ❌ `Co-authored-by` 行を省略する。
- ❌ 複数の異なる変更を一つのコミットに混ぜる。
- ❌ `gh` コマンドに `--no-pager` を付けようとする。
