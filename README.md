# dmf

## 概要
トーラス上の離散モース関数を図示します。

## 機能説明
Grid Sizeを変更するとトーラスの分割数が変更されます。初期値は3です。
Smoothを変更すると3Dのトーラスの滑らかさが変更されます。初期値は負荷軽減のため2です。
Color ScaleはPlotlyのcolorscaleに準拠しています。初期値はYlGnです。
ラジオボタンCell Name, DMF, No Labelを変更すると，ラベルが変更されます。3Dのトーラスについては、0-cellのCell Nameしか表示されません。初期値はCell Nameです。
チェックボックスShow Colorを有効にするとColor Scaleで着色されます。初期値はFalseです。
チェックボックスShow Arrowを有効にするとV-passが表示されます。初期値はFalseです。

## 必要なパッケージ
以下のパッケージが必要です。
パッケージのインストールには，`requirements.txt`を使用できます。
```bash
pip install -r requirements.txt
```
個別にインストールする場合は以下の通りです。
```bash
pip install dash numpy plotly
```
## その他
描画は環境によっては負荷がかかるため，分割数を制限しています。
変更する場合は634行目を変更してください。
```python
max=16, # 分割数を増やしたいとき変更
```
dashはデフォルトで8050番ポートを使用します。
8050番が使用できない場合は700行目を変更してください。
```python
app.run_server(port=8050) # port 8050を使用できない場合は変更
```

## ライセンス
このプロジェクトは、MIT Licenseのもとで公開されています。
