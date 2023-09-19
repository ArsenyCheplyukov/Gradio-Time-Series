import pandas as pd
from prophet import Prophet

import gradio as gr

demo = gr.Blocks()


def Dropdown_list(file):
    df = pd.read_csv(file.name)
    print(df.columns)
    return gr.Dropdown.update(choices=df.columns.tolist()), gr.Dropdown.update(
        choices=df.columns.tolist()
    )


def Prophet_work(file, x_axis_col, y_axis_col):
    staff_df = pd.read_csv(file.name, index_col=False)
    df = staff_df[[x_axis_col, y_axis_col]]
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df.dropna(inplace=True)

    m = Prophet()
    m.fit(df)

    duration = df["ds"].max() - df["ds"].min()

    units = {
        "1S": duration.total_seconds(),
        "60S": duration.total_seconds() // 60,
        "3600S": duration.total_seconds() // 3600,
        "D": duration.days,
        "W": duration.days // 7,
        "M": duration.days // 30,
        "Q": duration.days // 90,
        "Y": duration.days // 365,
    }

    result = {unit for unit, value in units.items() if value >= 100}

    min_unit = min(result, key=lambda unit: units[unit])
    min_value = int(units[min_unit] // 2)

    future = m.make_future_dataframe(periods=min_value, freq=min_unit)
    forecast = m.predict(future)
    fig = m.plot(forecast)
    return fig


options = [""]
with demo:
    file_input = gr.File(accept=".csv")
    b1 = gr.Button("Generate axis menu")

    x_axis = gr.Dropdown(options, label="Choose x axis")
    y_axis = gr.Dropdown(options, label="Choose y axis")
    b2 = gr.Button("Accept")

    output = gr.Plot()

    b1.click(Dropdown_list, inputs=file_input, outputs=[x_axis, y_axis])
    b2.click(Prophet_work, inputs=[file_input, x_axis, y_axis], outputs=output)


demo.launch(debug=True)
