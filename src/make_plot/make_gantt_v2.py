import plotly.graph_objects as go
import hashlib
from src.learner.common.Hyperparameters import *

class GanttChart:
    RULE_ABBR_MAP = {
        "SPTSDR": "SS",
        "SPTLDR": "SL",
        "LPTSDR": "LS",
        "LPTLDR": "LL",
    }

    @classmethod
    def add_custom_legend_box(cls, fig, family_color_map):
        # =========================
        # Legend 위치/크기 설정
        # =========================
        box_x0 = 0.840  # 박스 왼쪽 위치, 키우면 전체가 오른쪽 이동
        box_x1 = 0.995  # 박스 오른쪽 위치
        box_y0 = 0.28  # 박스 아래 위치
        box_y1 = 0.985  # 박스 위 위치

        pad_x = 0.008  # 박스 내부 왼쪽 여백
        text_x = box_x0 + pad_x

        title_font_size = 46
        item_font_size = 46

        # Family 색상 네모/텍스트 x 위치
        square_x0 = box_x0 + pad_x + 0.006
        square_x1 = square_x0 + 0.020
        family_text_x = square_x1 + 0.018

        # legend 박스
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=box_x0,
            y0=box_y0,
            x1=box_x1,
            y1=box_y1,
            fillcolor="rgba(255,255,255,0.85)",
            line=dict(color="rgba(0,0,0,0.25)", width=1),
            layer="above"
        )

        # =========================
        # Family type 제목 (위쪽)
        # =========================
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=text_x,
            y=0.975,
            text="<b>Family type</b>",
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
            font=dict(size=title_font_size, color="#2b3f5f")
        )

        # Family 항목 시작 위치
        start_y = 0.895
        gap = 0.065

        for idx, (family_type, color) in enumerate(family_color_map.items()):
            y = start_y - idx * gap

            # Family 색상 네모
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=square_x0,
                y0=y - 0.016,
                x1=square_x1,
                y1=y + 0.010,
                fillcolor=color,
                line=dict(color=color),
                layer="above"
            )

            # Family 텍스트
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=family_text_x,
                y=y,
                text=f"Family {family_type}",
                showarrow=False,
                align="left",
                xanchor="left",
                yanchor="middle",
                font=dict(size=item_font_size, color="#2b3f5f")
            )

        # =========================
        # Rule (아래쪽)
        # =========================
        rule_title_y = 0.595  # Rule 제목 위치
        rule_start_y = 0.510  # 첫 번째 Rule 항목 위치
        rule_gap = 0.065  # Rule 항목 사이 간격, 키우면 더 벌어짐

        # Rule 제목
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=text_x,
            y=rule_title_y,
            text="<b>Rule</b>",
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
            font=dict(size=title_font_size, color="#2b3f5f")
        )

        # Rule 항목들
        rule_items = [
            ("SS", "SPTSDR"),
            ("SL", "SPTLDR"),
            ("LS", "LPTSDR"),
            ("LL", "LPTLDR"),
        ]

        for idx, (abbr, full_name) in enumerate(rule_items):
            y = rule_start_y - idx * rule_gap

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=text_x,
                y=y,
                text=f"{abbr} = {full_name}",
                showarrow=False,
                align="left",
                xanchor="left",
                yanchor="middle",
                font=dict(size=item_font_size, color="#2b3f5f")
            )

    @classmethod
    def extract_rule_name(cls, rule):
        """
        예:
        4_SPTLDR -> SPTLDR
        SPTSDR -> SPTSDR
        """
        if rule is None:
            return ""
        rule = str(rule)
        return rule.split("_")[-1]

    @classmethod
    def abbreviate_rule(cls, rule):
        rule_name = cls.extract_rule_name(rule)
        return cls.RULE_ABBR_MAP.get(rule_name, rule_name)

    @classmethod
    def add_custom_legend(cls, fig, family_color_map):
        rule_items = [
            ("SS", "SPTSDR"),
            ("SL", "SPTLDR"),
            ("LS", "LPTSDR"),
            ("LL", "LPTLDR"),
        ]

        for idx, (abbr, full_name) in enumerate(rule_items):
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=0, color="rgba(0,0,0,0)"),
                name=f"{abbr} = {full_name}",
                legendgroup="rule",
                legendgrouptitle_text="Rule" if idx == 0 else None,
                showlegend=True,
                hoverinfo="skip"
            ))

        for idx, (family_type, color) in enumerate(family_color_map.items()):
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=18, color=color, symbol="square"),
                name=f"Family {family_type}",
                legendgroup="family",
                legendgrouptitle_text="Family type" if idx == 0 else None,
                showlegend=True,
                hoverinfo="skip"
            ))

    @classmethod
    def modify_width(cls, bar, width):
        bar.width = width

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
                r = int(r * darkness_factor)
                g = int(g * darkness_factor)
                b = int(b * darkness_factor)
                dark_color = f'#{r:02x}{g:02x}{b:02x}'
            color_map[color_id] = dark_color
        return color_map

    @classmethod
    def load_gantt_pickle(cls, pickle_name: str):
        """
        running_data_save_directory 에서 pickle_name을 불러와
        gantt_history 및 메타데이터 복원
        """
        Parameters.set_absolute_path()
        running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"
        pickle_path = os.path.join(running_data_save_directory, pickle_name)
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Gantt pickle not found: {pickle_path}")
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        # 필수 데이터 복원
        cls.gantt_history = data["gantt_history"]
        cls.dataset_id = data.get("dataset_id", "REPLAY")
        fitness = data.get("fitness", None)
        score = data.get("score", None)
        phase = data.get("phase", "replay")
        return fitness, score, phase

    @classmethod
    def main_gantt(cls, fitness, score, phase, show_custom_legend=True):
        cls.gantt_history['Resource_numeric'] = cls.gantt_history['Resource'].str.extract('(\d+)').astype(int)
        cls.gantt_history = cls.gantt_history.sort_values(by='Resource_numeric')
        cls.gantt_history = cls.to_bottom_setup_df(cls.gantt_history)

        cls.gantt_history['Color_ID'] = cls.gantt_history['Type'] + "_" + cls.gantt_history['Step'].astype(str) + "_" + \
                                        cls.gantt_history['family_type'].astype(str)
        color_map = cls.generate_color_map(cls.gantt_history['Color_ID'])
        cls.gantt_history['Color'] = cls.gantt_history['Color_ID'].map(color_map)

        cls.gantt_history['Job_Name'] = cls.gantt_history['Type'].apply(lambda x: x.split('_')[0])
        cls.gantt_history['Start'] = cls.gantt_history['Start'].astype(float)
        cls.gantt_history['Finish'] = cls.gantt_history['Finish'].astype(float)

        cls.gantt_history['Tardiness'] = cls.gantt_history['Finish'] - cls.gantt_history['due_date']
        cls.gantt_history["Rule_Name"] = cls.gantt_history["Rule"].apply(cls.extract_rule_name)
        cls.gantt_history["Rule_Abbr"] = cls.gantt_history["Rule"].apply(cls.abbreviate_rule)

        fig6 = go.Figure()
        time.sleep(0.1)

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

        for i, trace in enumerate(fig6.data):
            time.sleep(0.1)
            resource = trace.y[0]  # Resource 값은 y 배열의 첫 번째 값에 있음
            task_type = trace.name  # 현재 trace의 Type
            matching_row = cls.gantt_history[
                (cls.gantt_history['Resource'] == resource) & (cls.gantt_history['Color_ID'] == task_type)]
            if 'setup' in matching_row.iloc[0]['Type']:
                trace.customdata = [[matching_row.iloc[0]['Type'], matching_row.iloc[0]['p_time']]]
                trace.texttemplate = "%{customdata[0]}"
                trace.hovertemplate = None
            elif "Maintenance" in matching_row.iloc[0]['Type']:
                if Hyperparameters.action_type == 'dsp_rule' or Hyperparameters.action_type == 'dsp_rule_per_machine':
                    trace.customdata = [
                        [matching_row.iloc[0]['Type'], matching_row.iloc[0]['Rule'], matching_row.iloc[0]['p_time']]]
                    trace.texttemplate = "PM"
                    trace.hovertemplate = None
                else:
                    trace.customdata = [[matching_row.iloc[0]['Type'], matching_row.iloc[0]['p_time']]]
                    trace.texttemplate = "PM"
                    trace.hovertemplate = None
            else:
                trace.customdata = [[matching_row.iloc[0]['Type'],
                                     matching_row.iloc[0]['job'].deterioration_rate,
                                     matching_row.iloc[0]['family_type'],
                                     matching_row.iloc[0]['Resource'],
                                     matching_row.iloc[0]['job'].machine_quality,
                                     matching_row.iloc[0]['job'].previous_p_time,
                                     matching_row.iloc[0]['p_time'],
                                     matching_row.iloc[0]['Start'],
                                     matching_row.iloc[0]['Finish'],
                                     matching_row.iloc[0]['Rule_Abbr'],
                                     matching_row.iloc[0]['Rule_Name']]]

                trace.texttemplate = "%{customdata[9]}"

                trace.hovertemplate = (
                    "<span style='font-size:30px;'>"
                    "Rule=%{customdata[10]} (%{customdata[9]})<br><br><br>"
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
            trace.textfont = dict(size=34, color='white')
            trace.textposition = "inside"
            trace.insidetextanchor = "middle"

        for a, bar in enumerate(fig6.data):
            group_name = bar.name.split('_')[0]
            bar.legendgroup = group_name
            bar.showlegend = False
            # if a == 0 or all(b.legendgroup != group_name for b in fig6.data[:a]):
            #     bar.name = group_name  # 첫 번째 항목만 그룹 이름으로 표시
            #     bar.showlegend = True
            # else:
            #     bar.name = ""  # 나머지는 빈 문자열로 설정
            #     bar.showlegend = False  # 범례 표시 제거

        family_color_map = (
            cls.gantt_history[
                ~cls.gantt_history["Type"].str.contains("setup|Maintenance", case=False, na=False)
            ]
            .drop_duplicates("family_type")
            .sort_values("family_type")
            .set_index("family_type")["Color"]
            .to_dict()
        )
        if show_custom_legend:
            cls.add_custom_legend_box(fig6, family_color_map)

        fig6.update_layout(
            xaxis=dict(title="Time", titlefont=dict(size=54), tickfont=dict(size=48), range=[0, None], tick0=0,
                       dtick=100),
            yaxis=dict(title="Machine ID", titlefont=dict(size=54), tickfont=dict(size=48)),
            margin=dict(l=10, r=10, t=10, b=10),
            barmode='stack',
            autosize=True,
        )
        fig6.update_xaxes(
            # range=[0, cls.gantt_history['Finish'].max()],
            range=[0, 1200],
            constrain='domain'
        )

        for a, bar in enumerate(fig6.data):
            if 'setup' in bar.legendgroup:
                cls.modify_width(bar, 0.5)
            elif 'Maintenance' in bar.legendgroup:
                cls.modify_width(bar, 0.7)

        save_gantt = f"{pathConfig.running_data_path}{os.sep}"
        os.makedirs(save_gantt, exist_ok=True)
        fig6.write_html(
            f"{save_gantt}{Parameters.simulation_time}{cls.dataset_id}_F{Hyperparameters.family_type}"
            f"_{phase}_{fitness}_{score:.2f}_gantt_v3.html")


fitness, score, phase = GanttChart.load_gantt_pickle(
    pickle_name="260707_225930_M10_J200_F5_P10-50_D5-10_Fgap_U1_1533result_agent2two_stage_dqn_rule_1019.079_-4618.40_gantt.pkl")

GanttChart.main_gantt(fitness, score, phase, show_custom_legend=True)