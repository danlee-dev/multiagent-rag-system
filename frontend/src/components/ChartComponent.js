"use client";
import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
  Filler,
  ScatterController,
  BubbleController,
  PolarAreaController,
  registerables,
} from "chart.js";

import {
  Line,
  Bar,
  Pie,
  Doughnut,
  Radar,
  PolarArea,
  Scatter,
  Bubble,
} from "react-chartjs-2";

// 모든 Chart.js 컴포넌트와 스케일 등록
ChartJS.register(...registerables);

// 고급 색상 팔레트
const colorPalettes = {
  modern: [
    "#4F46E5",
    "#7C3AED",
    "#EC4899",
    "#EF4444",
    "#F59E0B",
    "#10B981",
    "#06B6D4",
    "#8B5CF6",
    "#F97316",
    "#84CC16",
  ],
  professional: [
    "#1E293B",
    "#334155",
    "#475569",
    "#64748B",
    "#94A3B8",
    "#CBD5E1",
    "#E2E8F0",
    "#F1F5F9",
    "#F8FAFC",
    "#FFFFFF",
  ],
  warm: [
    "#DC2626",
    "#EA580C",
    "#D97706",
    "#CA8A04",
    "#65A30D",
    "#059669",
    "#0891B2",
    "#0284C7",
    "#2563EB",
    "#7C3AED",
  ],
  pastel: [
    "#FED7E2",
    "#FECACA",
    "#FED7AA",
    "#FEF3C7",
    "#D1FAE5",
    "#A7F3D0",
    "#99F6E4",
    "#BFDBFE",
    "#C7D2FE",
    "#E9D5FF",
  ],
  dark: [
    "#6366F1",
    "#8B5CF6",
    "#EC4899",
    "#F59E0B",
    "#10B981",
    "#06B6D4",
    "#EF4444",
    "#84CC16",
    "#F97316",
    "#3B82F6",
  ],
  vibrant: [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FECA57",
    "#FF9FF3",
    "#54A0FF",
    "#5F27CD",
    "#00D2D3",
    "#FF9F43",
  ],
  corporate: [
    "#2C3E50",
    "#34495E",
    "#7F8C8D",
    "#95A5A6",
    "#BDC3C7",
    "#3498DB",
    "#9B59B6",
    "#E74C3C",
    "#E67E22",
    "#F39C12",
  ],
  gradient: [
    "#667eea",
    "#764ba2",
    "#f093fb",
    "#f5576c",
    "#4facfe",
    "#00f2fe",
    "#43e97b",
    "#38f9d7",
    "#ffecd2",
    "#fcb69f",
  ],
};


const convertBackendDataToChartData = (rawData, chartType) => {
  // 1. 이벤트 기반 데이터 (프로젝트 타임라인 등)
  if (rawData.events && Array.isArray(rawData.events)) {
    const events = rawData.events;

    if (chartType === "timeline" || chartType === "timeseries") {
      return {
        labels: events.map((event) => event.date || event.time || event.label),
        datasets: [
          {
            label: "프로젝트 진행도",
            data: events.map((event, index) => event.value || event.progress || index + 1),
            borderColor: "#4F46E5",
            backgroundColor: "#4F46E520",
            tension: 0.4,
            pointBackgroundColor: events.map((_, index) => {
              const colors = ["#DC2626", "#EA580C", "#D97706", "#10B981"];
              return colors[index % colors.length];
            }),
            pointBorderColor: "#FFFFFF",
            pointBorderWidth: 3,
            pointRadius: 8,
          },
        ],
      };
    }

    if (chartType === "gantt") {
      return {
        labels: events.map((event) => event.task || event.event || event.label),
        datasets: [
          {
            label: "작업 기간",
            data: events.map((event) => event.duration || event.value || 1),
            backgroundColor: events.map((_, index) => {
              const colors = ["#4F46E5", "#7C3AED", "#EC4899", "#10B981", "#F59E0B"];
              return colors[index % colors.length];
            }),
          },
        ],
      };
    }

    if (chartType === "bar" || chartType === "column") {
      return {
        labels: events.map((event) => event.event || event.label || event.name),
        datasets: [
          {
            label: "프로젝트 일정",
            data: events.map((event, index) => event.value || event.count || index + 1),
            backgroundColor: ["#4F46E5", "#7C3AED", "#EC4899", "#10B981"],
          },
        ],
      };
    }
  }

  // 2. 메트릭 기반 데이터 (성능 지표, KPI 등)
  if (rawData.metrics && Array.isArray(rawData.metrics)) {
    const metrics = rawData.metrics;

    if (chartType === "radar") {
      return {
        labels: metrics.map(m => m.name || m.label),
        datasets: [
          {
            label: "성능 지표",
            data: metrics.map(m => m.value || m.score),
            borderColor: "#4F46E5",
            backgroundColor: "#4F46E520",
            pointBackgroundColor: "#4F46E5",
          },
        ],
      };
    }

    if (chartType === "pie" || chartType === "doughnut") {
      return {
        labels: metrics.map(m => m.name || m.label),
        datasets: [
          {
            data: metrics.map(m => m.value || m.percentage),
            backgroundColor: colorPalettes.modern.slice(0, metrics.length),
          },
        ],
      };
    }
  }

  // 3. 시계열 데이터 (매출, 트래픽 등)
  if (rawData.timeseries && Array.isArray(rawData.timeseries)) {
    const timeseries = rawData.timeseries;

    return {
      labels: timeseries.map(t => t.date || t.time || t.period),
      datasets: [
        {
          label: rawData.label || "시계열 데이터",
          data: timeseries.map(t => t.value || t.amount),
          borderColor: "#4F46E5",
          backgroundColor: "#4F46E520",
          tension: 0.4,
        },
      ],
    };
  }

  // 4. 카테고리별 데이터 (부서별, 지역별 등)
  if (rawData.categories && Array.isArray(rawData.categories)) {
    const categories = rawData.categories;

    if (chartType === "stacked" && rawData.series) {
      return {
        labels: categories.map(c => c.name || c.label),
        datasets: rawData.series.map((series, index) => ({
          label: series.name || `시리즈 ${index + 1}`,
          data: categories.map(c => c.values ? c.values[index] || 0 : c.value || 0),
          backgroundColor: colorPalettes.modern[index % colorPalettes.modern.length],
          stack: "Stack 0",
        })),
      };
    }

    return {
      labels: categories.map(c => c.name || c.label),
      datasets: [
        {
          label: rawData.label || "카테고리 데이터",
          data: categories.map(c => c.value || c.count),
          backgroundColor: colorPalettes.modern.slice(0, categories.length),
        },
      ],
    };
  }

  // 5. 스캐터/버블 차트용 좌표 데이터
  if (rawData.points && Array.isArray(rawData.points)) {
    const points = rawData.points;

    if (chartType === "bubble") {
      return {
        datasets: [
          {
            label: rawData.label || "버블 차트",
            data: points.map(p => ({
              x: p.x,
              y: p.y,
              r: p.size || p.r || 10,
            })),
            backgroundColor: colorPalettes.modern.slice(0, points.length),
          },
        ],
      };
    }

    if (chartType === "scatter") {
      return {
        datasets: [
          {
            label: rawData.label || "스캐터 차트",
            data: points.map(p => ({
              x: p.x,
              y: p.y,
            })),
            backgroundColor: "#4F46E5",
            borderColor: "#4F46E5",
          },
        ],
      };
    }
  }

  // 6. 복합 차트 데이터 (여러 타입 조합)
  if (rawData.mixed && Array.isArray(rawData.mixed)) {
    const mixed = rawData.mixed;

    return {
      labels: mixed[0]?.labels || [],
      datasets: mixed.map((dataset, index) => ({
        type: dataset.type || (index === 0 ? 'bar' : 'line'),
        label: dataset.label || `데이터셋 ${index + 1}`,
        data: dataset.data || [],
        backgroundColor: dataset.backgroundColor || colorPalettes.modern[index % colorPalettes.modern.length],
        borderColor: dataset.borderColor || colorPalettes.modern[index % colorPalettes.modern.length],
        yAxisID: dataset.yAxisID || (index === 0 ? 'y' : 'y1'),
      })),
    };
  }

  // 7. 원시 숫자 배열 데이터
  if (Array.isArray(rawData) && rawData.length > 0 && typeof rawData[0] === 'number') {
    return {
      labels: rawData.map((_, index) => `항목 ${index + 1}`),
      datasets: [
        {
          label: "데이터",
          data: rawData,
          backgroundColor: colorPalettes.modern.slice(0, rawData.length),
        },
      ],
    };
  }

  // 8. 객체 배열 데이터 (일반적인 형태)
  if (Array.isArray(rawData) && rawData.length > 0 && typeof rawData[0] === 'object') {
    const firstItem = rawData[0];
    const labelKey = Object.keys(firstItem).find(key =>
      ['name', 'label', 'category', 'date', 'time'].includes(key)
    ) || Object.keys(firstItem)[0];

    const valueKey = Object.keys(firstItem).find(key =>
      ['value', 'count', 'amount', 'score', 'percentage'].includes(key)
    ) || Object.keys(firstItem)[1] || Object.keys(firstItem)[0];

    return {
      labels: rawData.map(item => item[labelKey]),
      datasets: [
        {
          label: valueKey || "데이터",
          data: rawData.map(item => item[valueKey]),
          backgroundColor: colorPalettes.modern.slice(0, rawData.length),
        },
      ],
    };
  }

  // 9. Chart.js 표준 형태는 그대로 반환
  if (rawData.labels && rawData.datasets) {
    return rawData;
  }

  // 10. 기본값
  return {
    labels: [],
    datasets: [],
  };
};


const getAdvancedAnimations = (chartType) => {
  return {
    duration: 150,
    easing: "linear",
  };
};


const chartComponents = {
  line: Line,
  bar: Bar,
  pie: Pie,
  doughnut: Doughnut,
  radar: Radar,
  polararea: PolarArea,
  scatter: Scatter,
  bubble: Bubble,
  area: Line, // fill 옵션으로 구분
  column: Bar,
  donut: Doughnut,
  polar: PolarArea,
  horizontalbar: Bar,
  stacked: Bar,
  mixed: Line, // 복합 차트는 Line 컴포넌트 사용
  funnel: Bar,
  waterfall: Bar,
  gauge: Doughnut,
  timeseries: Line,
  timeline: Line,
  gantt: Bar,
  // 새로운 차트 타입들
  multiline: Line, // 다중 라인 차트
  groupedbar: Bar, // 그룹화된 바 차트
  stackedarea: Line, // 스택된 영역 차트
  combo: Line, // 콤보 차트 (바+라인)
  heatmap: Bar, // 히트맵 (바 차트로 구현)
  treemap: Bar, // 트리맵 (바 차트로 구현)
  sankey: Bar, // 산키 다이어그램 (바 차트로 근사)
  candlestick: Line, // 캔들스틱 (라인 차트로 근사)
  violin: Bar, // 바이올린 플롯 (바 차트로 근사)
  boxplot: Bar, // 박스 플롯 (바 차트로 근사)
};

// 고급 옵션 설정
const getAdvancedOptions = (chartType, chartConfig) => {
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    devicePixelRatio: 2,
    animation: getAdvancedAnimations(chartType),
    plugins: {
      title: {
        display: true,
        text: chartConfig.title || "차트",
        font: {
          size: 24,
          weight: "600",
          family:
            "'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif",
        },
        color: "#1F2937",
        padding: {
          top: 20,
          bottom: 30,
        },
      },
      legend: {
        position: "top",
        align: "end",
        labels: {
          usePointStyle: true,
          pointStyle: "circle",
          padding: 20,
          font: {
            size: 13,
            weight: "500",
            family: "'Inter', sans-serif",
          },
          color: "#374151",
          generateLabels: function (chart) {
            let original;
            try {
              const chartType = chart.config.type;
              const overrides = ChartJS.overrides[chartType];
              if (
                overrides &&
                overrides.plugins &&
                overrides.plugins.legend &&
                overrides.plugins.legend.labels
              ) {
                original = overrides.plugins.legend.labels.generateLabels;
              }
            } catch (error) {
              console.warn("generateLabels 함수를 찾을 수 없습니다:", error);
            }


            if (!original || typeof original !== "function") {
              original = ChartJS.defaults.plugins.legend.labels.generateLabels;
            }

            const labels = original.call(this, chart);


            if (
              [
                "pie",
                "doughnut",
                "donut",
                "gauge",
                "polarArea",
                "polar",
              ].includes(chartType)
            ) {
              labels.forEach((label, index) => {
                if (
                  chart.data.datasets[0] &&
                  chart.data.datasets[0].data[index] !== undefined
                ) {
                  const value = chart.data.datasets[0].data[index];
                  label.text = `${label.text}: ${value}${
                    chartConfig.unit || ""
                  }`;
                }
              });
            }

            return labels;
          },
        },
      },
      tooltip: {
        enabled: true,
        backgroundColor: "rgba(17, 24, 39, 0.95)",
        titleColor: "#F9FAFB",
        bodyColor: "#F3F4F6",
        borderColor: "#6B7280",
        borderWidth: 1,
        cornerRadius: 12,
        displayColors: true,
        padding: 16,
        titleFont: {
          size: 14,
          weight: "600",
        },
        bodyFont: {
          size: 13,
          weight: "400",
        },
        filter: function (tooltipItem, data) {
          return tooltipItem.parsed !== null;
        },
        callbacks: {
          title: function (tooltipItems) {
            return tooltipItems[0].label || "";
          },
          label: function (context) {
            let label = context.dataset.label || "";
            if (label) {
              label += ": ";
            }
            label += new Intl.NumberFormat("ko-KR").format(
              context.parsed.y || context.parsed
            );
            if (chartConfig.unit) {
              label += chartConfig.unit;
            }
            return label;
          },
          afterLabel: function (context) {
            if (
              chartType === "pie" ||
              chartType === "doughnut" ||
              chartType === "donut" ||
              chartType === "gauge"
            ) {
              const dataset = context.dataset;
              const total = dataset.data.reduce((a, b) => a + b, 0);
              const percentage = ((context.parsed / total) * 100).toFixed(1);
              return `비율: ${percentage}%`;
            }
            return "";
          },
        },
      },
    },
    interaction: {
      intersect: false,
      mode: "index",
    },
    hover: {
      animationDuration: 100, // 300ms -> 100ms로 단축
    },
  };

  // 차트 타입별 특수 옵션
  switch (chartType) {
    case "line":
    case "area":
    case "timeseries":
    case "timeline":
      return {
        ...baseOptions,
        scales: {
          x: {
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
              drawBorder: false,
              tickColor: "transparent",
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
              padding: 12,
              maxRotation: 0,
            },
            border: {
              display: false,
            },
          },
          y: {
            beginAtZero: true,
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
              drawBorder: false,
              tickColor: "transparent",
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
              padding: 12,
              callback: function (value) {
                return new Intl.NumberFormat("ko-KR").format(value);
              },
            },
            border: {
              display: false,
            },
          },
        },
        elements: {
          line: {
            tension: 0.4,
            borderWidth: 3,
            borderCapStyle: "round",
            borderJoinStyle: "round",
          },
          point: {
            radius: 6,
            hoverRadius: 8,
            borderWidth: 3,
            backgroundColor: "#FFFFFF",
            hoverBorderWidth: 4,
          },
        },
      };

    case "bar":
    case "column":
    case "horizontalBar":
    case "stacked":
    case "funnel":
    case "waterfall":
    case "gantt":
      return {
        ...baseOptions,
        scales: {
          x: {
            grid: {
              display: chartType === "horizontalBar" || chartType === "gantt",
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
              padding: 12,
              maxRotation:
                chartType === "horizontalBar" || chartType === "gantt" ? 0 : 45,
            },
            border: {
              display: false,
            },
          },
          y: {
            beginAtZero: true,
            grid: {
              display: chartType !== "horizontalBar" && chartType !== "gantt",
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
              drawBorder: false,
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
              padding: 12,
              callback: function (value) {
                return new Intl.NumberFormat("ko-KR").format(value);
              },
            },
            border: {
              display: false,
            },
          },
        },
        indexAxis:
          chartType === "horizontalBar" || chartType === "gantt" ? "y" : "x",
        elements: {
          bar: {
            borderRadius: {
              topLeft: 8,
              topRight: 8,
              bottomLeft: 0,
              bottomRight: 0,
            },
            borderSkipped: false,
            borderWidth: 0,
          },
        },
      };

    case "pie":
    case "doughnut":
    case "donut":
    case "gauge":
      return {
        ...baseOptions,
        cutout:
          chartType === "doughnut" ||
          chartType === "donut" ||
          chartType === "gauge"
            ? "65%"
            : 0,
        circumference: chartType === "gauge" ? 180 : 360,
        rotation: chartType === "gauge" ? 270 : 0,
        plugins: {
          ...baseOptions.plugins,
          legend: {
            ...baseOptions.plugins.legend,
            position: "right",
            labels: {
              ...baseOptions.plugins.legend.labels,
              padding: 25,
              boxWidth: 15,
              boxHeight: 15,
            },
          },
        },
        elements: {
          arc: {
            borderWidth: 4,
            borderColor: "#FFFFFF",
            hoverBorderWidth: 6,
            borderAlign: "inner",
          },
        },
      };

    case "radar":
      return {
        ...baseOptions,
        scales: {
          r: {
            beginAtZero: true,
            grid: {
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
            },
            pointLabels: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#374151",
            },
            ticks: {
              display: false,
            },
          },
        },
        elements: {
          line: {
            borderWidth: 3,
            tension: 0.1,
          },
          point: {
            radius: 4,
            hoverRadius: 6,
            borderWidth: 2,
          },
        },
      };

    case "scatter":
    case "bubble":
      return {
        ...baseOptions,
        scales: {
          x: {
            type: "linear",
            position: "bottom",
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
          y: {
            beginAtZero: true,
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
        },
        elements: {
          point: {
            radius: 6,
            hoverRadius: 8,
            borderWidth: 2,
          },
        },
      };

    case "polarArea":
    case "polar":
      return {
        ...baseOptions,
        scales: {
          r: {
            beginAtZero: true,
            grid: {
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
            },
            pointLabels: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#374151",
            },
          },
        },
      };

    case "mixed":
    case "combo":
    case "multiline":
      return {
        ...baseOptions,
        scales: {
          x: {
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
              drawBorder: false,
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
              padding: 12,
            },
          },
          y: {
            type: "linear",
            display: true,
            position: "left",
            beginAtZero: true,
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
          y1: {
            type: "linear",
            display: true,
            position: "right",
            beginAtZero: true,
            grid: {
              drawOnChartArea: false,
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
        },
      };

    case "stackedarea":
      return {
        ...baseOptions,
        scales: {
          x: {
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
          y: {
            stacked: true,
            beginAtZero: true,
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
        },
        interaction: {
          mode: "index",
          intersect: false,
        },
      };

    case "groupedbar":
      return {
        ...baseOptions,
        scales: {
          x: {
            grid: {
              display: false,
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
              maxRotation: 45,
            },
          },
          y: {
            beginAtZero: true,
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
              lineWidth: 1,
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
        },
        plugins: {
          ...baseOptions.plugins,
          legend: {
            ...baseOptions.plugins.legend,
            position: "top",
          },
        },
      };

    case "heatmap":
      return {
        ...baseOptions,
        scales: {
          x: {
            type: "category",
            grid: {
              display: false,
            },
            ticks: {
              font: {
                size: 11,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
          y: {
            type: "category",
            grid: {
              display: false,
            },
            ticks: {
              font: {
                size: 11,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
        },
        elements: {
          bar: {
            borderWidth: 1,
            borderColor: "#FFFFFF",
          },
        },
      };

    case "candlestick":
      return {
        ...baseOptions,
        scales: {
          x: {
            type: "time",
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
          y: {
            beginAtZero: false,
            grid: {
              display: true,
              color: "rgba(156, 163, 175, 0.3)",
            },
            ticks: {
              font: {
                size: 12,
                weight: "500",
                family: "'Inter', sans-serif",
              },
              color: "#6B7280",
            },
          },
        },
      };

    default:
      return baseOptions;
  }
};


const preprocessAdvancedChartData = (chartType, data) => {
  if (!data || typeof data !== "object") {
    return { labels: [], datasets: [] };
  }

  // 백엔드 데이터 변환
  const convertedData = convertBackendDataToChartData(data, chartType);

  if (!Array.isArray(convertedData.datasets)) {
    return { labels: convertedData.labels || [], datasets: [] };
  }

  let processedData = { ...convertedData };

  // 차트 설정에서 색상 팔레트 선택 (기본값: modern)
  const paletteType = chartConfig.palette || "modern";
  const palette = colorPalettes[paletteType] || colorPalettes.modern;

  processedData.datasets = processedData.datasets.map(
    (dataset, datasetIndex) => {
      const baseColor = palette[datasetIndex % palette.length];

      switch (chartType) {
        case "line":
        case "timeseries":
        case "timeline":
          return {
            ...dataset,
            borderColor: baseColor,
            backgroundColor: `${baseColor}20`,
            pointBackgroundColor: "#FFFFFF",
            pointBorderColor: baseColor,
            pointBorderWidth: 3,
            pointRadius: 6,
            pointHoverRadius: 8,
            fill: false,
            tension: 0.4,
          };

        case "area":
          return {
            ...dataset,
            borderColor: baseColor,
            backgroundColor: `${baseColor}40`,
            pointBackgroundColor: "#FFFFFF",
            pointBorderColor: baseColor,
            pointBorderWidth: 3,
            fill: true,
            tension: 0.4,
          };

        case "bar":
        case "column":
        case "horizontalBar":
        case "funnel":
        case "waterfall":
          return {
            ...dataset,
            backgroundColor: palette.slice(0, dataset.data.length),
            borderColor: palette
              .slice(0, dataset.data.length)
              .map((color) => `${color}CC`),
            borderWidth: 2,
            borderRadius:
              chartType === "horizontalBar"
                ? {
                    topRight: 8,
                    bottomRight: 8,
                    topLeft: 0,
                    bottomLeft: 0,
                  }
                : {
                    topLeft: 8,
                    topRight: 8,
                    bottomLeft: 0,
                    bottomRight: 0,
                  },
            borderSkipped: false,
          };

        case "stacked":
          return {
            ...dataset,
            backgroundColor: palette[datasetIndex % palette.length],
            borderColor: `${palette[datasetIndex % palette.length]}CC`,
            borderWidth: 2,
            borderRadius: 4,
            stack: "Stack 0",
          };

        case "pie":
        case "doughnut":
        case "donut":
        case "gauge":
          return {
            ...dataset,
            backgroundColor: palette.slice(0, dataset.data.length),
            borderColor: "#FFFFFF",
            borderWidth: 4,
            hoverBorderWidth: 6,
            hoverBackgroundColor: palette
              .slice(0, dataset.data.length)
              .map((color) => `${color}DD`),
            ...(chartType === "gauge" && {
              circumference: 180,
              rotation: 270,
            }),
          };

        case "radar":
          return {
            ...dataset,
            borderColor: baseColor,
            backgroundColor: `${baseColor}30`,
            pointBackgroundColor: baseColor,
            pointBorderColor: "#FFFFFF",
            pointBorderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
          };

        case "scatter":
        case "bubble":
          return {
            ...dataset,
            backgroundColor: `${baseColor}80`,
            borderColor: baseColor,
            borderWidth: 2,
            pointRadius: 6,
            pointHoverRadius: 8,
          };

        case "polarArea":
        case "polar":
          return {
            ...dataset,
            backgroundColor: palette
              .slice(0, dataset.data.length)
              .map((color) => `${color}60`),
            borderColor: palette.slice(0, dataset.data.length),
            borderWidth: 2,
          };

        case "mixed":
        case "combo":
          return {
            ...dataset,
            type: dataset.type || (datasetIndex === 0 ? 'bar' : 'line'),
            backgroundColor: dataset.type === 'line' ? `${baseColor}20` : baseColor,
            borderColor: baseColor,
            borderWidth: dataset.type === 'line' ? 3 : 0,
            fill: dataset.type === 'line' ? false : undefined,
            tension: dataset.type === 'line' ? 0.4 : undefined,
            yAxisID: dataset.yAxisID || (datasetIndex === 0 ? 'y' : 'y1'),
          };

        case "multiline":
          return {
            ...dataset,
            borderColor: baseColor,
            backgroundColor: `${baseColor}20`,
            fill: false,
            tension: 0.4,
            borderWidth: 3,
            pointBackgroundColor: "#FFFFFF",
            pointBorderColor: baseColor,
            pointBorderWidth: 2,
            pointRadius: 4,
          };

        case "stackedarea":
          return {
            ...dataset,
            borderColor: baseColor,
            backgroundColor: `${baseColor}60`,
            fill: true,
            tension: 0.4,
            borderWidth: 2,
          };

        case "groupedbar":
          return {
            ...dataset,
            backgroundColor: baseColor,
            borderColor: `${baseColor}CC`,
            borderWidth: 1,
            borderRadius: 4,
          };

        case "heatmap":
          // 히트맵의 경우 데이터 값에 따라 색상 강도 조절
          return {
            ...dataset,
            backgroundColor: dataset.data.map((value, index) => {
              const maxValue = Math.max(...dataset.data);
              const intensity = value / maxValue;
              return `${baseColor}${Math.floor(intensity * 255).toString(16).padStart(2, '0')}`;
            }),
            borderColor: "#FFFFFF",
            borderWidth: 1,
          };

        case "candlestick":
          // 캔들스틱은 상승/하락에 따라 색상 구분
          return {
            ...dataset,
            borderColor: dataset.data.map((candle, index) =>
              candle.close > candle.open ? "#10B981" : "#EF4444"
            ),
            backgroundColor: dataset.data.map((candle, index) =>
              candle.close > candle.open ? "#10B98120" : "#EF444420"
            ),
            borderWidth: 2,
          };

        case "treemap":
        case "sankey":
          return {
            ...dataset,
            backgroundColor: palette.slice(0, dataset.data.length),
            borderColor: "#FFFFFF",
            borderWidth: 2,
          };

        case "violin":
        case "boxplot":
          return {
            ...dataset,
            backgroundColor: `${baseColor}40`,
            borderColor: baseColor,
            borderWidth: 2,
            borderRadius: 0,
          };

        default:
          return {
            ...dataset,
            backgroundColor: baseColor,
            borderColor: baseColor,
          };
      }
    }
  );

  // funnel 차트의 경우 데이터 정렬
  if (
    chartType === "funnel" &&
    processedData.datasets[0] &&
    Array.isArray(processedData.datasets[0].data)
  ) {
    const dataset = processedData.datasets[0];
    const sortedIndices = dataset.data
      .map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value)
      .map((item) => item.index);

    processedData.labels = sortedIndices.map((i) => processedData.labels[i]);
    processedData.datasets[0].data = sortedIndices.map((i) => dataset.data[i]);
    if (dataset.backgroundColor && Array.isArray(dataset.backgroundColor)) {
      processedData.datasets[0].backgroundColor = sortedIndices.map(
        (i) => dataset.backgroundColor[i]
      );
    }
  }

  // waterfall 차트의 경우 누적 계산
  if (
    chartType === "waterfall" &&
    processedData.datasets[0] &&
    Array.isArray(processedData.datasets[0].data)
  ) {
    const dataset = processedData.datasets[0];
    let cumulative = 0;
    const waterfallData = dataset.data.map((value, index) => {
      if (index === 0) {
        cumulative = value;
        return [0, value];
      } else {
        const start = cumulative;
        cumulative += value;
        return [start, cumulative];
      }
    });

    processedData.datasets[0].data = waterfallData;
  }

  // heatmap 차트의 경우 2D 데이터를 1D로 변환
  if (chartType === "heatmap" && data.matrix && Array.isArray(data.matrix)) {
    const matrix = data.matrix;
    const flatData = [];
    const labels = [];

    matrix.forEach((row, rowIndex) => {
      row.forEach((value, colIndex) => {
        flatData.push(value);
        labels.push(`${rowIndex}-${colIndex}`);
      });
    });

    processedData = {
      labels: labels,
      datasets: [
        {
          data: flatData,
          backgroundColor: flatData.map(value => {
            const maxValue = Math.max(...flatData);
            const intensity = value / maxValue;
            return `rgba(79, 70, 229, ${intensity})`;
          }),
        },
      ],
    };
  }

  // candlestick 차트의 경우 OHLC 데이터 처리
  if (chartType === "candlestick" && data.ohlc && Array.isArray(data.ohlc)) {
    const ohlc = data.ohlc;

    processedData = {
      labels: ohlc.map(candle => candle.date || candle.time),
      datasets: [
        {
          label: "Price",
          data: ohlc.map(candle => ({
            x: candle.date || candle.time,
            y: [candle.open, candle.high, candle.low, candle.close],
          })),
          borderColor: ohlc.map(candle =>
            candle.close > candle.open ? "#10B981" : "#EF4444"
          ),
          backgroundColor: ohlc.map(candle =>
            candle.close > candle.open ? "#10B98140" : "#EF444440"
          ),
        },
      ],
    };
  }

  return processedData;
};

// 백엔드에서 받은 차트 설정(JSON)을 props로 받음
export function ChartComponent({ chartConfig }) {
  const chartId = React.useMemo(() => {
    return `${chartConfig?.type || "unknown"}-${
      chartConfig?.title || "untitled"
    }-${Date.now()}-${Math.random()}`;
  }, [chartConfig?.type, chartConfig?.title]);

  console.log(`ChartComponent 렌더링: ${chartId}`);
  console.log("받은 chartConfig:", chartConfig);
  console.log("chartConfig.data:", chartConfig?.data);
  console.log("chartConfig.data.datasets:", chartConfig?.data?.datasets);

  // 입력 데이터 검증
  if (!chartConfig) {
    return (
      <div
        className="chart-error"
        style={{
          padding: "24px",
          textAlign: "center",
          color: "#DC2626",
          border: "2px solid #FCA5A5",
          borderRadius: "12px",
          backgroundColor: "#FEF2F2",
          fontFamily: "'Inter', sans-serif",
          fontSize: "14px",
          fontWeight: "500",
        }}
      >
        차트 설정이 없습니다.
      </div>
    );
  }

  if (!chartConfig.type) {
    return (
      <div
        className="chart-error"
        style={{
          padding: "24px",
          textAlign: "center",
          color: "#DC2626",
          border: "2px solid #FCA5A5",
          borderRadius: "12px",
          backgroundColor: "#FEF2F2",
          fontFamily: "'Inter', sans-serif",
          fontSize: "14px",
          fontWeight: "500",
        }}
      >
        차트 타입이 지정되지 않았습니다.
      </div>
    );
  }

  // 데이터 검증 강화 - 백엔드 데이터 형태도 허용
  if (!chartConfig.data) {
    console.error("chartConfig.data가 없습니다");
    return (
      <div
        className="chart-error"
        style={{
          padding: "24px",
          textAlign: "center",
          color: "#DC2626",
          border: "2px solid #FCA5A5",
          borderRadius: "12px",
          backgroundColor: "#FEF2F2",
          fontFamily: "'Inter', sans-serif",
          fontSize: "14px",
          fontWeight: "500",
        }}
      >
        차트 데이터가 없습니다.
      </div>
    );
  }

  // 백엔드 데이터 변환 시도
  const hasValidData =
    // Chart.js 표준 형태
    (chartConfig.data.datasets && Array.isArray(chartConfig.data.datasets)) ||
    // 백엔드 이벤트 데이터 형태
    (chartConfig.data.events && Array.isArray(chartConfig.data.events)) ||
    // 기타 백엔드 데이터 형태들
    Object.keys(chartConfig.data).length > 0;

  if (!hasValidData) {
    console.error("데이터 구조 오류:", {
      hasData: !!chartConfig.data,
      hasDatasets: !!chartConfig.data?.datasets,
      datasetsType: typeof chartConfig.data?.datasets,
      isArray: Array.isArray(chartConfig.data?.datasets),
      actualData: chartConfig.data,
    });

    return (
      <div
        className="chart-error"
        style={{
          padding: "24px",
          textAlign: "center",
          color: "#DC2626",
          border: "2px solid #FCA5A5",
          borderRadius: "12px",
          backgroundColor: "#FEF2F2",
          fontFamily: "'Inter', sans-serif",
          fontSize: "14px",
          fontWeight: "500",
        }}
      >
        차트 데이터가 올바르지 않습니다.
        <br />
        <small style={{ color: "#7F1D1D", fontSize: "12px" }}>
          유효한 데이터 형태가 아닙니다.
        </small>
        <br />
        <details style={{ marginTop: "10px", fontSize: "11px" }}>
          <summary>디버그 정보</summary>
          <pre
            style={{ textAlign: "left", overflow: "auto", maxHeight: "200px" }}
          >
            {JSON.stringify(chartConfig.data, null, 2)}
          </pre>
        </details>
      </div>
    );
  }

  const chartType = chartConfig.type.toLowerCase();
  const ChartTypeComponent = chartComponents[chartType];

  if (!ChartTypeComponent) {
    return (
      <div
        className="chart-error"
        style={{
          padding: "24px",
          textAlign: "center",
          color: "#D97706",
          border: "2px solid #FCD34D",
          borderRadius: "12px",
          backgroundColor: "#FFFBEB",
          fontFamily: "'Inter', sans-serif",
          fontSize: "14px",
          fontWeight: "500",
        }}
      >
        지원하지 않는 차트 타입입니다: {chartConfig.type}
        <br />
        <small style={{ color: "#92400E", fontSize: "12px" }}>
          지원 타입: line, bar, pie, doughnut, radar, polarArea, scatter,
          bubble, area, column, horizontalBar, stacked, funnel, waterfall,
          gauge, timeseries, timeline, gantt, mixed, combo, multiline,
          groupedbar, stackedarea, heatmap, treemap, sankey, candlestick,
          violin, boxplot
        </small>
      </div>
    );
  }

  // 데이터 전처리
  const processedData = preprocessAdvancedChartData(
    chartType,
    chartConfig.data
  );

  if (
    !processedData ||
    !Array.isArray(processedData.datasets) ||
    processedData.datasets.length === 0
  ) {
    return (
      <div
        className="chart-error"
        style={{
          padding: "24px",
          textAlign: "center",
          color: "#DC2626",
          border: "2px solid #FCA5A5",
          borderRadius: "12px",
          backgroundColor: "#FEF2F2",
          fontFamily: "'Inter', sans-serif",
          fontSize: "14px",
          fontWeight: "500",
        }}
      >
        차트 데이터 처리 중 오류가 발생했습니다.
        <br />
        <small style={{ color: "#7F1D1D", fontSize: "12px" }}>
          데이터셋이 비어있거나 유효하지 않습니다.
        </small>
      </div>
    );
  }


  const advancedOptions = getAdvancedOptions(chartType, chartConfig);
  const finalOptions = {
    ...advancedOptions,
    ...chartConfig.options,
    plugins: {
      ...advancedOptions.plugins,
      ...chartConfig.options?.plugins,
    },
    scales: {
      ...advancedOptions.scales,
      ...chartConfig.options?.scales,
    },
  };

  console.log("최종 처리된 데이터:", processedData);

  return (
    <div
      className="enhanced-chart-container"
      style={{
        position: "relative",
        height: "500px",
        width: "100%",
        padding: "24px",
        backgroundColor: "#FFFFFF",
        borderRadius: "16px",
        boxShadow:
          "0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
        border: "1px solid rgba(229, 231, 235, 0.8)",
        marginBottom: "32px",
        backdropFilter: "blur(8px)",
        fontFamily:
          "'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      <ChartTypeComponent
        options={finalOptions}
        data={processedData}
        key={chartId}
      />
    </div>
  );
}
