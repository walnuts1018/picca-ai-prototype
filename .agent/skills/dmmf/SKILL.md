---
name: dmmf
description: 関数型ドメインモデリングの原則を実践するためのガイドライン
when_to_use: アーキテクチャ設計、コード生成、リファクタリング、コードレビューなど、ドメイン駆動設計を関数型のパラダイムで実践する必要がある場合
---

# Skill: Functional Domain Modeling in Rust

## 目的 (Objective)

関数型プログラミングの概念とドメイン駆動設計（DDD）の原則を組み合わせ、Rustを用いてドメインモデルの意図を正確に反映し、安全で保守性の高いアーキテクチャを設計・実装する。ドメインの知識を静的な「型」に落とし込み、ソフトウェアの複雑性を管理するとともに、コード自体をドメインの実行可能なドキュメントにする。

## 1. コア原則とアーキテクチャ (Core Principles & Architecture)

- **ユビキタス言語の反映**: コード（特に型シグネチャ）は、ドメインの知識をそのまま表現するドキュメントとして機能しなければならない。開発者中心の用語ではなく、ドメインエキスパートの用語を使用する。
- **イベントとワークフローへの焦点**: 静的なデータ構造ではなく、ビジネスイベントとコマンド（システムで起こる事象）からモデリングを開始する。
- **垂直スライス (Vertical Slices)**: 従来の水平なレイヤー（Controller -> Service -> Repository）に分けるのではなく、ワークフローごとに独立した「垂直スライス」としてコードを構成する。各スライスには、入力からプロセス、出力イベントまで必要なコードをすべて含める。
- **I/Oとビジネスロジックの分離 (Push I/O to the edges)**:
- コアドメインの関数は副作用を持たない「純粋関数」として実装する。
- **I/Oサンドイッチ**: `[不純なI/Oによる読み込み] -> [純粋なビジネスロジック] -> [不純なI/Oによる書き込み]` というパターンを徹底し、データベースの読み書きなどの副作用はワークフローの最初と最後にのみ配置する。

- **CQRS (コマンド・クエリ責務分離)**: データストアから情報を読み取る「クエリ」と、状態を変更する「コマンド」は用途が異なるため、それぞれ専用の型とモジュールに分離する。
- **DTOの活用とシリアライズ**: 外部インフラストラクチャとの通信には、ドメインオブジェクトを直接シリアライズせず、プリミティブ型で構成された中間型（DTO）を使用する。DTOからドメインモデルへ変換する際にバリデーションを行う。

## 2. 型によるドメインモデリング (Type-Driven Modeling)

Rustの強力な代数的データ型（ADT）を活用して要件を型として定義し、コンパイル時に不正な状態を弾く（Make illegal states unrepresentable）。

- **単純な値（Newtypeパターン）**:
- `String`や`i32`などのプリミティブ型をビジネスロジックに直接露出させず（Primitive Obsessionの排除）、必ず単一要素のタプル構造体（ラッパー型）を作成する。
- スマートコンストラクタを実装し、インスタンス化の時点で制約（文字数や範囲など）をバリデーションして`Result`を返す。

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct WidgetCode(String);

impl WidgetCode {
    pub fn create(code: String) -> Result<Self, String> {
        if code.starts_with('W') && code.len() == 5 {
            Ok(WidgetCode(code))
        } else {
            Err("WidgetCode must start with 'W' and be 5 characters long".to_string())
        }
    }
}

```

- **複雑なデータの合成**:
- **ANDの合成 (直積型 / Struct)**: 密接に関連するデータのグループや、同時に必要なデータは構造体（`struct`）で表現する。
- **ORの合成 (直和型 / Enum)**: 異なる状態や選択肢のバリエーションは列挙型（`enum`）で表現する。Booleanフラグによる暗黙的な状態管理を排除し、状態遷移は状態ごとの個別の型を用意して`enum`で束ねる。

```rust
pub enum ProductCode {
    Widget(WidgetCode),
    Gizmo(GizmoCode),
}

```

- **省略可能な値とエラーの明示**:
- 値が欠損する可能性がある場合は`Option<T>`を使用する。
- エラーは汎用的な文字列ではなく、ドメインエラーを表す`enum`としてモデル化し、すべてのエラーケースを列挙する。

## 3. ワークフローのモデリングと関数合成 (Workflow & Function Composition)

ワークフローは、データの変換を行う「純粋関数」のパイプラインとして設計する。

- **入力と出力の明示**: ワークフローの各ステップは、「未検証のデータ（入力）」から「検証済みのデータ（出力）またはエラー」を返す関数として定義する。
- **全域関数 (Total Functions)**: 想定されるすべての入力に対して、有効な出力を返すように関数を設計する。
- **2トラックモデル (Result指向のエラーハンドリング)**:
- 失敗する可能性がある処理は例外を投げず、必ず`Result<T, E>`を使用して型シグネチャに「エフェクト」を明示する。
- `?`演算子や`and_then`、`map`、`map_err`を活用し、成功ルート（Okトラック）と失敗ルート（Errトラック）を透過的に結合したエレガントなパイプラインを構築する。

```rust
pub fn place_order(
    unvalidated_order: UnvalidatedOrder,
) -> Result<Vec<PlaceOrderEvent>, PlaceOrderError> {
    validate_order(unvalidated_order)         // Result<ValidatedOrder, Error>
        .and_then(price_order)                // Result<PricedOrder, Error>
        .map(acknowledge_order)               // PricedOrder -> AcknowledgedOrder
        .map(create_events)                   // AcknowledgedOrder -> Vec<PlaceOrderEvent>
}

```

## 4. 依存関係の注入 (Dependency Injection)

- **明示的な依存関係**: データベース操作や外部サービスへのアクセスなど、ワークフローが必要とする関数やサービスは暗黙的に呼び出さず、関数の引数（関数ポインタ `fn`、クロージャ `impl Fn`、トレイト `impl Trait`）として明示的に注入する。

```rust
pub fn validate_order<F1, F2>(
    check_product_exists: F1, // 依存関係1
    check_address_exists: F2, // 依存関係2
    unvalidated_order: UnvalidatedOrder, // 入力
) -> Result<ValidatedOrder, ValidationError>
where
    F1: Fn(&ProductCode) -> bool,
    F2: Fn(&UnvalidatedAddress) -> Result<CheckedAddress, AddressValidationError>,
{
    // 依存関係を利用した純粋なビジネスロジック
}

```

- **依存関係の隠蔽**: アプリケーションの最上位（コンポジションルート）で部分適用（クロージャによる変数のキャプチャなど）を利用して依存関係を解決し、下位のワークフロー（純粋なパイプライン）にはシンプルな型シグネチャの関数のみを渡す。

## 5. アシスタントへの指示 (Prompting Directives)

Rustのコードを生成またはレビューする際は、以下のルールを厳守すること：

1. **ManagerやHandlerを避ける**: `OrderManager`や`OrderHelper`のような開発者中心の言葉ではなく、ドメインの「ユビキタス言語」をそのまま型名や関数名に使用する。
2. **Primitive Obsessionの禁止**: 業務的な意味を持つ文字列や数値は、必ず`struct`を用いたNewtypeパターンでラップし、生成時にバリデーションを行う。
3. **I/Oのサンドイッチを徹底する**: ドメインロジックを担う関数内で、直接データベースにアクセスしたり現在時刻を取得したりしない。必要なデータは引数として渡し、結果は値として返す。
4. **型でドキュメント化する**: 関数のシグネチャを見ただけで、「何が入力され」「どんなエラーが起こり得るか」「何が出力されるか」が明確にわかるように、`Result`とドメイン固有の型を最大限に活用する。
