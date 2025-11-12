import streamlit as st


def page_3():
    st.write("page3です")

    # クエリーボタンを表示
    if st.button("クエリー"):
        print("hello")
        st.success("ターミナルに 'hello' を出力しました！")
