"""Export the evidence tables as a dependency-free OOXML workbook."""

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent / "results"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/package/2006/relationships"
C = "http://schemas.openxmlformats.org/package/2006/content-types"


def element(name, attrs=None, parent=None, ns=S):
    node = ET.Element(f"{{{ns}}}{name}", attrs or {})
    if parent is not None:
        parent.append(node)
    return node


def column(index):
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def sheet(rows, notes=False):
    root = element("worksheet")
    view = element("sheetView", {"workbookViewId": "0"}, element("sheetViews", parent=root))
    element("pane", {"ySplit": "1", "topLeftCell": "A2", "activePane": "bottomLeft", "state": "frozen"}, view)
    cols = element("cols", parent=root)
    for i in range(1, len(rows[0]) + 1):
        width = "100" if notes and i == 2 else "36" if i == 1 else "22"
        element("col", {"min": str(i), "max": str(i), "width": width, "customWidth": "1"}, cols)
    data = element("sheetData", parent=root)
    for ri, values in enumerate(rows, 1):
        height = "48" if ri == 1 else "58" if notes else "34"
        row = element("row", {"r": str(ri), "ht": height, "customHeight": "1"}, data)
        for ci, value in enumerate(values, 1):
            value = str(value)
            attrs = {"r": f"{column(ci)}{ri}", "s": "1" if ri == 1 else "0"}
            if ri > 1 and re.fullmatch(r"-?\d+(?:\.\d+)?", value):
                cell = element("c", attrs, row)
                element("v", parent=cell).text = value
            else:
                cell = element("c", {**attrs, "t": "inlineStr"}, row)
                element("t", parent=element("is", parent=cell)).text = value
    element("autoFilter", {"ref": f"A1:{column(len(rows[0]))}{len(rows)}"}, root)
    return root


def main():
    notes = [
        ["项目", "口径或结论"],
        ["日期", "2026-10-18，X99四张CMP170HX；本机漫剧两卡未参与"],
        ["单路decode", "选定配置逐次结果见Decode工作表；单路100尚未达到"],
        ["混合八并发", "端到端聚合逐次结果见Decode工作表；不能当作每路速度"],
        ["prefill", "服务端新计算输入token/请求prefill阶段墙钟；每档两次，前缀缓存命中0"],
        ["TTFT", "X99本机客户端发起请求至首个非空输出增量"],
        ["端到端", "输出token/整批总耗时，包含首字等待"],
        ["decode口径", "completion_tokens/(最后输出-首次输出)；计入首次增量token，略偏高"],
        ["完整性", "3000/1024 token压测因预算结束；故事/CSV脚本正常stop，代码仅AST语法验证"],
        ["第二轮实验", "动态缓存未证明更快，未采用，恢复固定缓存与基线学习状态"],
        ["第三轮实验", "查询3未证明更快，未采用；CMP拒绝CUPTI，没有GPU内核轨迹；恢复无profiler配置"],
        ["第四轮实验", "CUDA Event取得阶段区间，含流等待及CPU回调；固定核心A/B/A见附表，不把诊断请求速度当生产吞吐"],
        ["第五轮实验", "CPU回放与完整服务对照单列；预取和DSpark5均未证明提速，未采用；CPU耗时不等于模型token/s，单流80尚未达到"],
        ["第六轮实验", "GPU内核/主机中转筛查，无生产改动；显式线程参数才会应用blocks_per_sm；原配置复查不算提速"],
        ["第七轮实验", "常驻CPU调度桥接；隔离时延与真实专家逐位校验单列，模型性能以A/B/A完整服务为准"],
        ["详细报告", "同目录X99最终测试报告.md及upstream-evidence原始记录"],
    ]
    tables = [("指标说明", notes)]
    for title, filename in [
        ("Decode与并发", "X99最终结果-Decode与并发.csv"),
        ("Prefill逐次", "X99最终结果-Prefill.csv"),
        ("第二轮优化对照", "X99第二轮优化对照.csv"),
        ("第三轮优化对照", "X99第三轮优化对照.csv"),
        ("第四轮阶段计时", "X99第四轮阶段计时.csv"),
        ("第四轮固定核心对照", "X99第四轮固定核心对照.csv"),
        ("第五轮CPU回放", "X99第五轮CPU回放.csv"),
        ("第五轮服务对照", "X99第五轮服务对照.csv"),
        ("第五轮Prefill对照", "X99第五轮Prefill对照.csv"),
        ("第六轮GPU内核", "X99第六轮GPU内核.csv"),
        ("第六轮传输", "X99第六轮传输.csv"),
        ("第六轮服务复查", "X99第六轮服务复查.csv"),
        ("第六轮Prefill复查", "X99第六轮Prefill复查.csv"),
        ("第七轮分层诊断", "X99第七轮分层诊断.csv"),
        ("第七轮桥接隔离", "X99第七轮桥接隔离.csv"),
        ("第七轮真实专家校验", "X99第七轮真实专家校验.csv"),
        ("第七轮服务对照", "X99第七轮服务对照.csv"),
        ("第七轮Prefill复查", "X99第七轮Prefill复查.csv"),
    ]:
        with (ROOT / filename).open(encoding="utf-8-sig", newline="") as source:
            tables.append((title, list(csv.reader(source))))

    types = element("Types", ns=C)
    for extension, content in [("rels", "application/vnd.openxmlformats-package.relationships+xml"), ("xml", "application/xml")]:
        element("Default", {"Extension": extension, "ContentType": content}, types, C)
    rels = element("Relationships", ns=P)
    element("Relationship", {"Id": "rId1", "Type": R + "/officeDocument", "Target": "xl/workbook.xml"}, rels, P)
    book = element("workbook")
    sheets = element("sheets", parent=book)
    bookrels = element("Relationships", ns=P)
    parts = {"_rels/.rels": rels, "xl/workbook.xml": book, "xl/_rels/workbook.xml.rels": bookrels}
    for i, (title, rows) in enumerate(tables, 1):
        element("sheet", {"name": title, "sheetId": str(i), f"{{{R}}}id": f"rId{i}"}, sheets)
        element("Relationship", {"Id": f"rId{i}", "Type": R + "/worksheet", "Target": f"worksheets/sheet{i}.xml"}, bookrels, P)
        parts[f"xl/worksheets/sheet{i}.xml"] = sheet(rows, notes=i == 1)
    styles = ET.fromstring(f'''<styleSheet xmlns="{S}"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF24476A"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>''')
    parts["xl/styles.xml"] = styles
    element("Relationship", {"Id": f"rId{len(tables) + 1}", "Type": R + "/styles", "Target": "styles.xml"}, bookrels, P)
    for path in parts:
        if path.endswith(".rels"):
            continue
        kind = "worksheet" if "/worksheets/" in path else "styles" if path.endswith("styles.xml") else "sheet.main"
        element("Override", {"PartName": "/" + path, "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml." + kind + "+xml"}, types, C)
    parts["[Content_Types].xml"] = types
    destination = ROOT / "X99最终测试结果.xlsx"
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path, node in parts.items():
            archive.writestr(path, ET.tostring(node, encoding="utf-8", xml_declaration=True))
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        for path in archive.namelist():
            ET.fromstring(archive.read(path))
    print(destination)


if __name__ == "__main__":
    main()
