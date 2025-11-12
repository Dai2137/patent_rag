import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from infra.loader.other_loader import OtherLoaders
from infra.loader.st36_patent_loader import St36PatentLoader
from infra.loader.st96_patent_loader import St96PatentLoader
from infra.loader.st96_utility_loader import St96UtilityLoader
from model.patent import Patent


class CommonLoader:
    """
    ST36・ST96形式の特許・実用新案のXMLをロードし、Patentオブジェクトを生成するクラスです。
    """

    def __init__(self):
        """
        コンストラクタです。各種ローダを初期化します。
        """
        self.st36_patent_loader = St36PatentLoader()
        self.st96_patent_loader = St96PatentLoader()
        self.st96_utility_loader = St96UtilityLoader()
        self.other_loader = OtherLoaders()

    def run(self, path: Path | str) -> Patent:
        """
        XMLファイルのパスを受け取り、タグの種類に応じて適切なローダに処理を委譲し、Patentオブジェクトを生成します。
        """
        path = Path(path) # 原則、strではなくPathで持つ

        # if "JP2024524707A" in path.as_posix():
        #     return self.other_loader.load_JP2024524707A(path)

        tree: ET.ElementTree[ET.Element] = ET.parse(str(path))

        root: ET.Element | None = tree.getroot()
        if root is None:
            raise ValueError("rootが取得できません。")

        patent = self._root_2_patent(root, path)
        return patent

    def content_2_patent(self, xml_content: str):
        """
        ファイルパスがどうしても不明な場合は、XML文字列を直接渡してもよい。
        ただし、パスが不明だと後からその特許を参照できないので、非推奨です。
        """
        tree: ET.Element = ET.fromstring(xml_content)
        patent = self._root_2_patent(tree, path=None)
        return patent

    def _root_2_patent(self, root: ET.Element, path: Optional[Path] = None) -> Patent:
        """
        XMLのルート要素を受け取り、タグの種類に応じて適切なローダに処理を委譲し、Patentオブジェクトを生成します。
        """
        tag: str = root.tag

        if tag.endswith("jp-official-gazette"):
            patent = self.st36_patent_loader.run(root, path)
        elif tag.endswith(("UnexaminedPatentPublication", "RegisteredPatentPublication", "InternationalPatentPublication")):
            patent = self.st96_patent_loader.run(root, path)
        elif tag.endswith("RegisteredUtilityModelPublication"):
            patent = self.st96_utility_loader.run(root, path)
        else:
            raise ValueError(f"未定義のXMLスキーマです。タグ: {root.tag}")
        return patent


def save_json(patent: Patent, path: Path):
    import json
    from dataclasses import asdict

    payload = asdict(patent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


# 単体テスト
if __name__ == "__main__":
    paths = [
        # # JPO標準形式（A）（XMLタグ：jp-official-gazette）
        Path("data/result_1/0/JP2010000001A/text.txt"),
        Path("data/result_1/0/JP2015039043A/text.txt"),
        # # ST96特許形式（A）（XMLタグ：UnexaminedPatentPublication）
        Path("data/result_9/1/JP2022043358A/text.txt"),
        Path("data/result_9/1/JP2022008727A/text.txt"),  # <U>タグ
        Path("data/result_9/3/JP2022008730A/text.txt"),
        Path("data/result_9/3/JP2022036324A/text.txt"),
        # # ST96特許形式（A）（XMLタグ：InternationalPatentPublication）（PCT国際公開？）
        Path("data/result_13/0/JP2022514722A/text.txt"),
        Path("data/result_13/0/JP2022524171A/text.txt"),
        Path("data/result_13/1/JP2022519792A/text.txt"),
        Path("data/result_13/1/JP2022533491A/text.txt"),
        # # ST96実用新案形式（U）（XMLタグ：RegisteredUtilityModelPublication）
        Path("data/result_18/13/JP3236365U/text.txt"),
        Path("data/result_18/13/JP3236395U/text.txt"),
        Path("data/result_18/19/JP3250568U/text.txt"),
        # # ST96特許形式（B）（XMLタグ：RegisteredPatentPublication）
        Path("data/result_18/19/JP7550342B/text.txt"),
        Path("data/result_18/19/JP7559286B/text.txt"),
        Path("data/result_18/19/JP7646111B/text.txt"),
        # # 請求項ロード失敗ケース1. ST36 amendment
        Path("data/result_18/0/JP3214154U/text.txt"),
        Path("data/result_18/0/JP3214781U/text.txt"),
        Path("data/result_18/1/JP3215724U/text.txt"),  # 請求項はOK、明細書はNG（Amendment対応が必要）
        Path("data/result_18/1/JP3217077U/text.txt"),
        Path("data/result_18/5/JP3222705U/text.txt"),
        # # 請求項ロード失敗ケース2. ST36 1層目に画像
        Path("data/result_18/5/JPWO2018134950A1/text.txt"),
        # # 請求項ロード失敗ケース3. ST36 2層目に画像（例：表>画像、数式>画像）
        Path("data/result_5/0/JP2013540401A/text.txt"),
        Path("data/result_5/0/JP2013546219A/text.txt"),  # 明細書にも対応要
        # # "【請求" がない
        Path("data/result_18/19/JP3250096U/text.txt"),
        # # ST96 Amendment
        Path("data/result_15/13/JP2024125135A/text.txt"),  # 請求項はOK、明細書がNG
        Path("data/result_15/13/JP2024125136A/text.txt"),  # 請求項はOK、明細書がNG
        Path("data/result_15/13/JP2024125258A/text.txt"),
        Path("data/result_15/15/JP2024153521A/text.txt"),  # 請求項はOK、明細書がNG
        # # 請求項エラー（ラスト3件）
        Path("data/result_1/27/JP2011011021A/text.txt"),
        Path("data/result_1/33/JP2011067573A/text.txt"),
        Path("data/result_2/2/JP2011115514A/text.txt"),
        # # 要約書がないエラー
        Path("data/result_13/8/JP2022503667A/text.txt"),
        Path("data/result_13/8/JP2022508435A/text.txt"),
        Path("data/result_15/15/JP2023534040A/text.txt"),
        Path("data/result_15/15/JP2023534250A/text.txt"),
        # # 要約書がないエラーver2
        Path("data/result_5/5/JP2014502745A/text.txt"),  # PDFでも空欄だった。なす術なし。
        Path("data/result_12/10/JP2021516888A/text.txt"),  # 画像データ。OCRすればテキスト抽出できる。
        # # # ProxyError個別対応
        Path("data/result_16/11/JP2024524707A/text.txt"),
        # # <abstract>タグがない（古い特許Bには要約自体ない）
        Path("data/result_18/13/JP6976480B/text.txt"),
        Path("data/result_18/13/JP6982925B/text.txt"),
        # # 請求項の本文がない（Amendment対応）
        Path("data/result_18/12/JP3236424U/text.txt"),
        Path("data/result_18/17/JP3244015U/text.txt"),
    ]

    for path in paths:
        loader = CommonLoader()
        patent: Patent = loader.run(path)
        json_path = path.with_suffix(".json")
        save_json(patent, json_path)
