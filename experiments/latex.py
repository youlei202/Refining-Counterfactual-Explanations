import numpy as np
from experiments import labels


class TikzPlotGenerator:
    def __init__(self, data):
        self.data = data
        self.colors = [
            # "cyan",
            # "blue",
            # "orange",
            # "green",
            "red",
            "black",
            "black",
            # "magenta",
            # "yellow",
            # "purple",
            # "brown",
            # "black",
        ]
        self.line_styles = [
            "solid",
            "dashed",
            "solid",
            # "solid",
            # "solid",
            # "solid",
            # "solid",
            # "solid",
            # "solid",
            # "solid",
        ]

    def compute_statistics(self, y_lists):
        y_array = np.array(y_lists)
        y_mean = np.mean(y_array, axis=0)
        y_sem = np.std(y_array, axis=0) / np.sqrt(y_array.shape[0])
        confidence_level = 3.291  # for 99.9% confidence interval
        y_lower = y_mean - confidence_level * y_sem
        y_upper = y_mean + confidence_level * y_sem
        return y_mean, y_lower, y_upper

    def sanitize_name(self, name):
        return labels.mapping[name]

    def generate_plot_code(self, metric, methods, corrected_y_lists=None):
        plot_code = """
\\begin{tikzpicture}
\\begin{axis}[
    width=4.4cm, height=3.6cm,
    legend pos=north east,
    legend style={
        draw=none,
        % font=\\scriptsize,
        legend image code/.code={
            \draw[mark repeat=2,mark phase=2]
                plot coordinates {
                    (0cm,0cm)
                    (0.4cm,0cm) % Adjust the length here
                };
        },
        inner xsep=0pt, % Adjust inner x separation
        inner ysep=0pt  % Adjust inner y separation
    },
    legend cell align={left},
    grid=major,
    % title={SVM, $\\A_{\\text{CE}}=$GlobeCE},
    title style={font=\\footnotesize, yshift=-1ex}, 
    font=\\footnotesize,
    % Increase margin of x and y tick labels
    xticklabel style={xshift=-0pt, yshift=-3pt},
    yticklabel style={xshift=-3pt, yshift=0pt},
]
        """

        color_idx = 0

        for method, metrics in self.data.items():
            if method in methods and metric in metrics:
                groups = metrics[metric]
                x_list = groups[0]["x_list"]

                if method == "optimality":
                    y_lists = corrected_y_lists
                else:
                    y_lists = [group["y_list"] for group in groups]

                y_mean, y_lower, y_upper = self.compute_statistics(y_lists)

                color = self.colors[color_idx % len(self.colors)]
                line_style = self.line_styles[color_idx % len(self.line_styles)]
                color_idx += 1

                sanitized_method = self.sanitize_name(method)
                sanitized_metric = self.sanitize_name(metric)

                plot_code += f"\\addplot[name path={sanitized_method}-{sanitized_metric}-line, {color}, {line_style}] coordinates {{\n"
                for x, y in zip(x_list, y_mean):
                    plot_code += f"    ({x}, {y})\n"
                plot_code += "};\n"
                plot_code += f"%\\addlegendentry{{{sanitized_method}}}\n"

                if method in ["optimality", "CF_ExactMatch"]:
                    continue

                plot_code += f"\\addplot[name path={sanitized_method}-{sanitized_metric}-upper, {color}, opacity=0.3, forget plot] coordinates {{\n"
                for x, y in zip(x_list, y_upper):
                    plot_code += f"    ({x}, {y})\n"
                plot_code += "};\n"

                plot_code += f"\\addplot[name path={sanitized_method}-{sanitized_metric}-lower, {color}, opacity=0.3, forget plot] coordinates {{\n"
                for x, y in zip(x_list, y_lower):
                    plot_code += f"    ({x}, {y})\n"
                plot_code += "};\n"

                plot_code += f"\\addplot[fill opacity=0.3, {color}, forget plot] fill between[of={sanitized_method}-{sanitized_metric}-upper and {sanitized_method}-{sanitized_metric}-lower];\n"

        plot_code += "\\end{axis}\n"
        plot_code += "\\end{tikzpicture}\n"

        return plot_code
