
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import hashlib
import time
from src.common.Parameters import *
from src.learner.common.Hyperparameters import *

class GanttChart:
    gantt_on = Parameters.gantt_on
    HISTORY_COLUMNS = ['Type', 'JOB_ID', 'Task', 'Start', 'Finish', 'Resource', 'Rule', 'Step',
                       'Q_diff', 'Q_check', 'p_time', 'family_type', 'job_arrival_time', 'due_date', 'job']
    gantt_history = pd.DataFrame([], columns=HISTORY_COLUMNS)
    history_rows = []  # save_histories가 쌓는 행 (그릴 때 build_history에서 한 번에 DataFrame으로 변환)
    step = 0
    color_by_color_mapper = {'j01': 'rgb(255, 204, 204)', 'j02': 'rgb(229, 204, 255)',
                             'j03': 'rgb(255, 255, 204)', 'j04': 'rgb(204, 229, 255)',
                             'j05': 'rgb(145, 229, 204)'}
    dataset_id = ""
    @classmethod
    def reset(cls):
        cls.gantt_history = pd.DataFrame([], columns=cls.HISTORY_COLUMNS)
        cls.history_rows = []
        cls.step = 0

    @classmethod
    def save_histories(cls, event_type, job_id, jop, start, end, machine_id, rule, step, q_time_diff, q_time_check, p_time, family_type, job_arrival_time, due_date, job):
        cls.history_rows.append(dict(
            Type=event_type,
            JOB_ID=job_id,
            Task=jop,
            Start=start,
            Finish=end,
            Resource=machine_id,
            Rule=rule,
            Step=step,
            Q_diff=q_time_diff,
            Q_check=q_time_check,
            p_time=p_time,
            family_type=family_type,
            job_arrival_time=job_arrival_time,
            due_date=due_date,
            job=job
        ))  # 간트차트를 위한 딕셔너리 생성 (DataFrame 변환은 build_history에서)
        cls.step += 1

    @classmethod
    def build_history(cls):
        """쌓아 둔 행으로 gantt_history DataFrame 생성 (행마다 .loc로 추가하면 O(n^2)이라 그릴 때 한 번에 만듦)"""
        if cls.history_rows:
            cls.gantt_history = pd.DataFrame(cls.history_rows, columns=cls.HISTORY_COLUMNS)
            cls.history_rows = []

    @classmethod
    def update_df(cls):
        step_rule = []
        for i in range(len(cls.gantt_history)):
            if str(cls.gantt_history["Rule"].loc[i]) != "None":
                step_rule.append(str(cls.gantt_history["Step"].loc[i]) + "-" + str(cls.gantt_history["Rule"].loc[i]))
            else:
                step_rule.append("NONE")
        cls.gantt_history["Step-Rule"] = step_rule

        id_op = []
        for i in range(len(cls.gantt_history)):
            if str(cls.gantt_history["Task"].loc[i]) != "None":
                id_op.append(str(cls.gantt_history["JOB_ID"].loc[i]) + "-" + str(cls.gantt_history["Task"].loc[i]))
            else:
                id_op.append("NONE")
        cls.gantt_history["ID_OP"] = id_op

    @classmethod
    def play_gantt(cls, datasetId, fitness, score, phase):
        cls.build_history()
        cls.update_df()
        cls.dataset_id = datasetId
        for i in cls.gantt_on:
            if cls.gantt_on[i]:
                if i == "main_gantt":
                    cls.main_gantt(fitness, score, phase)
                elif i == "machine_on_job_number":
                    cls.mahicne_on_job_number()
                elif i == "machine_gantt":
                    cls.machine_gantt()
                elif i == "DSP_gantt":
                    cls.DSP_gantt()
                elif i == "step_DSP_gantt":
                    cls.step_DSP_gantt()
                elif i == "heatMap_gantt":
                    cls.heatMap_gantt()
                elif i == "job_gantt_for_Q_time":
                    cls.job_gantt_for_Q_time()

    @classmethod
    def mahicne_on_job_number(cls):
        fig = px.bar(cls.gantt_history, x="Resource", y="Type", color="Type", facet_row="Type")
        fig.update_yaxes(matches=None)
        fig.show()

    @classmethod
    def machine_gantt(cls):
    # fig,write_html(f"{PathInfo.xlsx}{os.sep}temp_target.html", default_width=2300, default_height=900)
        plotlydf3 = cls.gantt_history.sort_values(by=['Type'], ascending=True)
        fig2 = px.timeline(plotlydf3, x_start="Start", x_end="Finish", y="Type", template="seaborn", color="Resource",
                           text="Resource", width=2000, height=1000)
        fig2.update_traces(marker=dict(line_color="yellow", cmid=1000))
        fig2.show()

    @classmethod
    def DSP_gantt(cls):
        fig3 = px.timeline(cls.gantt_history, x_start="Start", x_end="Finish", y="Resource", template="simple_white", color="Type",
                           color_discrete_sequence=px.colors.qualitative.Dark24, text="Rule", width=2000, height=800)
        [(cls.modify_width(bar, 0.7), cls.modify_text(bar)) for bar in fig3.data if ('setup' in bar.legendgroup)]
        fig3.show()

    @classmethod
    def step_DSP_gantt(cls):
        cls.gantt_history = cls.gantt_history.sort_values(by='Resource')
        y_category_order = cls.gantt_history['Resource'].unique().tolist()

        fig4 = px.timeline(cls.gantt_history, x_start="Start", x_end="Finish", y="Resource", template="simple_white", color="Type",
                           color_discrete_sequence=px.colors.qualitative.Dark24, text="Step-Rule", width=2000, height=800,
                           category_orders = {"Resource": y_category_order})
        if cls.color_by_color_mapper:
            for color_mapper, color in cls.color_by_color_mapper.items():
                [cls.modify_bar(bar, color=color) for bar in fig4.data if bar.name == color_mapper]
        #[(cls.modify_width(bar, 0.7), cls.modify_text(bar))
         #for bar in fig4.data if ('setup' in bar.legendgroup)]
        fig4.show()

    @classmethod
    def modify_bar(cls, bar, color=None):
        """
        Bar 속성 지정 메서드
         - color: Bar의 색상
         - opacity: Bar의 투명도 (0 ~ 1)
         - line_width: Bar 테두리 굵기 (0 ~ 1.6)
        """
        if color:
            cls.modify_color(bar, color)

    @classmethod
    def modify_color(cls, bar, color):
        """
        Bar의 색상을 지정할 수 있다.
        """
        bar.marker['color'] = color

    @classmethod
    def heatMap_gantt(cls):
        fig5 = px.timeline(cls.gantt_history, x_start="Start", x_end="Finish", y="Rule", template="simple_white", color="Rule",
                           color_discrete_sequence=px.colors.qualitative.Dark24, text="Step-Rule", width=2000, height=800)
        fig5.show()

    @classmethod
    def generate_color_map(cls, color_ids, darkness_factor=0.8):
        color_map = {}
        for color_id in color_ids:
            job_type = color_id.split("_")[0]  # Color_ID에서 Type 부분만 추출
            family_type = color_id.split("_")[-1]
            if job_type == "setup":
                dark_color = '#808080'
            else:
                if family_type == 'NONE':
                    hash_object = hashlib.md5(job_type.encode())
                    hex_color = hash_object.hexdigest()[:6]  # 6자리 색상 코드 생성
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                else:
                    family_type = int(family_type)
                    r = (family_type * 123 + 45) % 256  # R 값 생성
                    g = (family_type * 67 + 89) % 256  # G 값 생성
                    b = (family_type * 89 + 123) % 256  # B 값 생성
                    # r = (family_type * 80 + 100) % 256
                    # g = (family_type * 160 + 150) % 256
                    # b = (family_type * 240 + 200) % 256

                r = int(r * darkness_factor)
                g = int(g * darkness_factor)
                b = int(b * darkness_factor)
                dark_color = f'#{r:02x}{g:02x}{b:02x}'
            color_map[color_id] = dark_color
        return color_map

    @classmethod
    def main_gantt(cls, fitness, score, phase):
        cls.gantt_history['Resource_numeric'] = cls.gantt_history['Resource'].str.extract('(\d+)').astype(int)
        cls.gantt_history = cls.gantt_history.sort_values(by='Resource_numeric')
        cls.gantt_history = cls.to_bottom_setup_df(cls.gantt_history)

        cls.gantt_history['Color_ID'] = cls.gantt_history['Type'] + "_" + cls.gantt_history['Step'].astype(str) + "_" + cls.gantt_history['family_type'].astype(str)
        color_map = cls.generate_color_map(cls.gantt_history['Color_ID'])
        cls.gantt_history['Color'] = cls.gantt_history['Color_ID'].map(color_map)

        cls.gantt_history['Job_Name'] = cls.gantt_history['Type'].apply(lambda x: x.split('_')[0])
        cls.gantt_history['Start'] = cls.gantt_history['Start'].astype(float)
        cls.gantt_history['Finish'] = cls.gantt_history['Finish'].astype(float)

        cls.gantt_history['Tardiness'] = cls.gantt_history['Finish'] - cls.gantt_history['due_date']

        save_dir = f"{pathConfig.running_data_path}{os.sep}"
        os.makedirs(save_dir, exist_ok=True)
        pickle_path = (
            f"{save_dir}"
            f"{Parameters.simulation_time}_{cls.dataset_id}_{phase}"
            f"{Hyperparameters.action_type}_{fitness}_{score:.2f}_gantt.pkl"
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(
                {
                    "gantt_history": cls.gantt_history,
                    "dataset_id": cls.dataset_id,
                    "fitness": fitness,
                    "score": score,
                    "phase": phase,
                },
                f, protocol=pickle.HIGHEST_PROTOCOL
            )

        fig6 = go.Figure()
        tardiness = False
        if Hyperparameters.object == 'makespan':
            tardiness = False
            object = 'Makespan'
        else:
            tardiness = True
            object = 'Total Tardiness'
        for i, row in cls.gantt_history.iterrows():
            fig6.add_trace(go.Bar(
                x=[row['p_time']],  # 막대의 길이 (p_time 사용)
                y=[row['Resource']],
                base=row['Start'],  # 막대의 시작 위치
                name=row['Color_ID'],
                orientation='h',
                text=f"{row['Job_Name']} - {row['p_time']}",
                hovertemplate=(
                    f"Type={row['Type']}<br>Family_type={row['family_type']}<br>"
                    f"p_time={row['p_time']:.3f}<br>Start={row['Start']:.3f}<br>"
                    f"Finish={row['Finish']:.3f}<br>Resource={row['Resource']}<br>"
                    if 'setup' not in row['Type'] else None
                ),
                marker=dict(color=row['Color'])  # 지정된 색상을 사용
            ))
            if tardiness:
                if row['Type'] == 'setup' or row['Type'] == 'NOTHING' or row['Type'] == "Maintenance":
                    pass
                else:
                    if row['Start'] > row['due_date']:  # 시작이 DUE DATE 이후인 경우
                        fig6.add_shape(
                            type="rect",
                            x0=row['Start'],
                            x1=row['Finish'],
                            y0=row['Resource_numeric'] - 1 - 0.4,
                            y1=row['Resource_numeric'] - 1  + 0.4,
                            layer="above",
                            line=dict(color="black", width=4)  # 파란색 테두리와 굵기 설정
                        )
                    elif row['Finish'] > row['due_date']:  # 끝난 시간이 DUE DATE 이후인 경우
                        fig6.add_shape(
                            type="rect",
                            x0=row['due_date'],  # DUE DATE 이후 구간만
                            x1=row['Finish'],
                            y0=row['Resource_numeric'] - 1 - 0.4,
                            y1=row['Resource_numeric'] - 1 + 0.4,
                            layer="above",
                            line=dict(color="black", width=4)  # 파란색 테두리와 굵기 설정
                        )
        # (Resource, Color_ID) -> 처음 나오는 행의 위치 (막대마다 전체 행을 다시 검색하지 않도록 한 번에 만듦)
        first_row_pos = {}
        for pos, key in enumerate(zip(cls.gantt_history['Resource'], cls.gantt_history['Color_ID'])):
            first_row_pos.setdefault(key, pos)
        for i, trace in enumerate(fig6.data):
            resource = trace.y[0]  # Resource 값은 y 배열의 첫 번째 값에 있음
            task_type = trace.name  # 현재 trace의 Type
            matching_row = cls.gantt_history.iloc[first_row_pos[(resource, task_type)]]  # 조건에 맞는 첫 행
            if 'setup' in matching_row['Type']:
                trace.customdata = [[matching_row['Type'], matching_row['p_time']]]
                trace.texttemplate = "%{customdata[0]}"
                trace.hovertemplate = None
            elif "Maintenance" in matching_row['Type']:
                if Hyperparameters.action_type == 'dsp_rule' or Hyperparameters.action_type == 'dsp_rule_per_machine':
                    trace.customdata = [[matching_row['Type'], matching_row['Rule'], matching_row['p_time']]]
                    trace.texttemplate = "PM"
                    trace.hovertemplate = None
                else:
                    trace.customdata = [[matching_row['Type'], matching_row['p_time']]]
                    trace.texttemplate = "PM"
                    trace.hovertemplate = None
            elif 'NOTHING' in matching_row['Type']:
                trace.customdata = [[matching_row['Type'], matching_row['Resource'],
                                     matching_row['p_time'], matching_row['Start'], matching_row['Finish']]]
                trace.texttemplate = (
                    "%{customdata[0]}<br>"
                    "P: %{customdata[2]:.3f}"
                )
                trace.hovertemplate = (
                    "<span style='font-size:30px;'>"
                    "Type=%{customdata[0]}<br><br><br>"
                    "Machine=%{customdata[1]}<br><br><br>"
                    "p_time=%{customdata[2]:.3f}<br><br><br>"
                    "Start=%{customdata[3]:.3f}<br><br><br>"
                    "Finish=%{customdata[4]:.3f}<br><br><br>"
                    "</span>"
                )
            else:
                if (Hyperparameters.action_type == 'dsp_rule' or Hyperparameters.action_type == 'dsp_rule_per_machine'
                        or Hyperparameters.action_type == "two_stage_dqn_rule"):
                    trace.customdata = [[matching_row['Type'], matching_row['job'].deterioration_rate,
                                         matching_row['family_type'], matching_row['Resource'],
                                         matching_row['job'].machine_quality,
                                         matching_row['job'].previous_p_time,
                                         matching_row['p_time'], matching_row['Start'],
                                         matching_row['Finish'], matching_row['Rule']]]
                    trace.texttemplate = (
                        "%{customdata[9]}"
                    )
                    trace.hovertemplate = (
                        "<span style='font-size:30px;'>"
                        "Rule=%{customdata[9]}<br><br><br>"
                        "Type=%{customdata[0]}<br><br><br>"
                        "deterioration_rate=%{customdata[1]:.3f}<br><br><br>"
                        "Family_type=%{customdata[2]}<br><br><br>"
                        "Machine=%{customdata[3]}<br><br><br>"
                        "Machine Quality=%{customdata[4]:.3f}<br><br><br>"
                        "previous_p_time=%{customdata[5]:.3f}<br><br><br>"
                        "p_time=%{customdata[6]:.3f}<br><br><br>"
                        "Start=%{customdata[7]:.3f}<br><br><br>"
                        "Finish=%{customdata[8]:.3f}<br><br><br>"
                        "</span>"
                    )
                else:
                    trace.customdata = [[matching_row['Type'], matching_row['job'].deterioration_rate,
                                         matching_row['family_type'], matching_row['Resource'],
                                         matching_row['job'].machine_quality, matching_row['job'].previous_p_time,
                                         matching_row['p_time'], matching_row['Start'], matching_row['Finish'],
                                         matching_row['job_arrival_time'], matching_row['due_date']]]
                    trace.texttemplate = (
                        "%{customdata[0]}"
                    )
                    trace.hovertemplate = (
                        "<span style='font-size:30px;'>"
                        "Type=%{customdata[0]}<br><br><br>"
                        "deterioration_rate=%{customdata[1]:.3f}<br><br><br>"
                        "Family_type=%{customdata[2]}<br><br><br>"
                        "Machine=%{customdata[3]}<br><br><br>"
                        "Machine Quality=%{customdata[4]:.3f}<br><br><br>"
                        "previous_p_time=%{customdata[5]:.3f}<br><br><br>"
                        "p_time=%{customdata[6]:.3f}<br><br><br>"
                        "Start=%{customdata[7]:.3f}<br><br><br>"
                        "Finish=%{customdata[8]:.3f}<br><br><br>"
                        "</span>"
                    )
            trace.textfont = dict(size=26, color='white')
            trace.textposition = "inside"
            trace.insidetextanchor = "middle"

        seen_groups = set()  # 이미 범례에 나온 그룹 (앞 막대 전체를 다시 훑지 않도록)
        for bar in fig6.data:
            group_name = bar.name.split('_')[0]
            bar.legendgroup = group_name
            if group_name not in seen_groups:
                seen_groups.add(group_name)
                bar.name = group_name  # 첫 번째 항목만 그룹 이름으로 표시
                bar.showlegend = True
            else:
                bar.name = ""  # 나머지는 빈 문자열로 설정
                bar.showlegend = False  # 범례 표시 제거

        fig6.update_layout(
            title=dict(
                text=f"{object}: {fitness}, Cumulative Reward: {score:.3f}",
                font=dict(size=40),
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis=dict(title="Time", titlefont=dict(size=24), tickfont=dict(size=20), range=[0, None]),
            yaxis=dict(title="Machine", titlefont=dict(size=24), tickfont=dict(size=24)),
            margin=dict(l=50, r=50, t=100, b=50),  # 좌, 우, 상, 하 여백 설정
            legend=dict(
                font=dict(size=30),  # 글씨 크기 키우기
                title_text="Job",
                title_font=dict(size=30),  # 범례 제목 크기
                traceorder="normal"  # 정렬된 데이터를 사용
            ),
            barmode='stack',  # 막대가 옆으로 나열되도록 설정
            autosize=True,
        )
        fig6.update_xaxes(tick0=0, dtick=50)

        for a, bar in enumerate(fig6.data):
            if 'setup' in bar.legendgroup:
                cls.modify_width(bar, 0.5)
            elif 'Maintenance' in bar.legendgroup:
                cls.modify_width(bar, 0.7)
        save_gantt = f"{pathConfig.simulator_result_path}{os.sep}"
        if Parameters.gantt_on_check:
            os.makedirs(save_gantt, exist_ok=True)
        if phase == None:
            fig6.write_html(f"{save_gantt}{Parameters.simulation_time}_{cls.dataset_id}_{Hyperparameters.action_type}_{fitness}_{score:.2f}_gantt.html")
        elif phase == "GA":
            fig6.write_html(f"{save_gantt}{Parameters.simulation_time}_{cls.dataset_id}_F{Hyperparameters.family_type}_{fitness}_{score:.2f}_GA_gantt.html")
        elif phase == 'run':
            fig6.write_html(f"{save_gantt}{Parameters.simulation_time}_{cls.dataset_id}_F{Hyperparameters.family_type}_{phase}_PM{Hyperparameters.maintenance_time}_{Hyperparameters.maintenance_threshold}_{fitness}_{score:.2f}_gantt.html")
        elif Hyperparameters.action_type == "two_stage_rnn":
            fig6.write_html(f"{save_gantt}{Parameters.simulation_time}_{cls.dataset_id}_F{Hyperparameters.family_type}_{phase}_PM{Hyperparameters.maintenance_time}_{fitness}_{score:.2f}_gantt.html")
        elif phase == "dsp_rule":
            fig6.write_html(f"{save_gantt}{Parameters.simulation_time}_{cls.dataset_id}_{phase}_{fitness}_{score:.2f}_gantt.html")
        else:
            fig6.write_html(f"{save_gantt}{Parameters.simulation_time}_{cls.dataset_id}_F{Hyperparameters.family_type}_A{Hyperparameters.rule_count}_{phase}_PM{Hyperparameters.maintenance_time}_{fitness}_{score:.2f}_gantt.html")

    @classmethod
    def job_gantt_for_Q_time(cls):
        df = cls.gantt_history_arrival_and_due._append(cls.gantt_history, ignore_index=True)
        df = df.sort_values(by=['Start', "Finish"], ascending=[False, False])
        df = cls.to_top_arrival_df(df)
        df = cls.to_bottom_due_df(df)
        fig8 = px.timeline(df, x_start="Start", x_end="Finish", y="JOB_ID", template="simple_white", color="Q_check",
                           color_discrete_sequence=px.colors.qualitative.Dark24, text="Q_diff", width=2000, height=2000)
        [(cls.modify_width(bar, 0.7), cls.modify_text(bar))
         for bar in fig8.data if ('setup' in bar.legendgroup)]
        fig8.show()

    @classmethod
    def modify_width(cls, bar, width):
        """
        막대의 너비를 설정합니다.
        width = (단위 px)
        """
        bar.width = width

    @classmethod
    def modify_text(cls, bar):
        """
        막대의 텍스트를 설정합니다.
        width = (단위 px)
        """
        bar.text = "su"

    @classmethod
    def to_top_arrival_df(cls, df):
        """
        figure의 경우 위에서 부터 bar 생성됩니다.
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """
        arrival_df = df.loc[df['Type'] == 'job_arrival']
        df = df[df['Type'] != 'job_arrival']
        arrival_df = arrival_df._append(df, ignore_index=True)
        return arrival_df

    @classmethod
    def to_bottom_setup_df(cls, df):
        """
        figure의 경우 위에서 부터 bar 생성됩니다.
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """
        setup_df = df.loc[df['Type'] == 'setup']
        df = df[df['Type'] != 'setup']
        df = df._append(setup_df, ignore_index=True)
        return df

    @classmethod
    def to_bottom_due_df(cls, df):
        """
        figure의 경우 위에서 부터 bar 생성됩니다.
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """
        setup_df = df.loc[df['Type'] == 'due_date']
        df = df[df['Type'] != 'due_date']
        df = df._append(setup_df, ignore_index=True)
        return df