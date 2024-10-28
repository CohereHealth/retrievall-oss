import io
import pytest
from pyarrow.csv import read_csv
from retrievall.ocr import corpus_from_tesseract_table


@pytest.fixture
def tesseract_table():
    """
    An example Tesseract-style table.

    See here for more details on tabular Tessaract outputs:
    https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html#tsv-output
    """
    inline_csv = (
        "level,page_num,block_num,par_num,line_num,word_num,left,top,width,height,conf,text\n"
        "1,1,0,0,0,0,0,0,300,400,-1,\n"
        "2,1,1,0,0,0,20,20,110,90,-1,\n"
        "3,1,1,1,0,0,20,20,180,30,-1,\n"
        "4,1,1,1,1,0,20,20,110,10,-1,\n"
        "5,1,1,1,1,1,20,20,30,10,96.063751,The\n"
        "5,1,1,1,1,2,60,20,50,10,95.965691,(quick)\n"
        "4,1,1,1,2,0,20,40,200,10,-1,\n"
        "5,1,1,1,2,1,20,40,70,10,95.835831,[brown]\n"
        "5,1,1,1,2,2,100,40,30,10,94.899742,fox\n"
        "5,1,1,1,2,3,140,40,60,10,96.683357,jumps!\n"
        "3,1,1,2,0,0,20,80,90,30,-1,\n"
        "4,1,1,2,1,0,20,80,80,10,-1,\n"
        "5,1,1,2,1,1,20,80,40,10,96.912064,Over\n"
        "5,1,1,2,1,2,40,80,30,10,96.887390,the\n"
        "4,1,1,2,2,0,20,100,100,10,-1,\n"
        "5,1,1,2,2,1,20,100,60,10,90.893219,<lazy>\n"
        "5,1,1,2,2,2,90,100,30,10,96.538940,dog\n"
        "1,2,0,0,0,0,0,0,300,400,-1,\n"
        "2,2,1,0,0,0,20,20,110,90,-1,\n"
        "3,2,1,1,0,0,20,20,180,30,-1,\n"
        "4,2,1,1,1,0,20,20,110,10,-1,\n"
        "5,2,1,1,1,1,20,20,30,10,96.063751,The\n"
        "5,2,1,1,1,2,60,20,50,10,95.965691,~groovy\n"
        "4,2,1,1,2,0,20,40,200,10,-1,\n"
        "5,2,1,1,2,1,20,40,70,10,95.835831,minute!\n"
        "5,2,1,1,2,2,100,40,30,10,94.899742,dog\n"
        "5,2,1,1,2,3,140,40,60,10,96.683357,bounds\n"
        "3,2,1,2,0,0,20,80,90,30,-1,\n"
        "4,2,1,2,1,0,20,80,80,10,-1,\n"
        "5,2,1,2,1,1,20,80,40,10,96.912064,UPON\n"
        "5,2,1,2,1,2,40,80,30,10,96.887390,the\n"
        "4,2,1,2,2,0,20,100,100,10,-1,\n"
        "5,2,1,2,2,1,20,100,60,10,90.893219,sleepy\n"
        "5,2,1,2,2,2,90,100,30,10,96.538940,fox\n"
    )

    table = read_csv(io.BytesIO(inline_csv.encode()))

    return table


@pytest.fixture
def ocr_corpus(tesseract_table):
    return corpus_from_tesseract_table(tesseract_table, document_id="abc123")
