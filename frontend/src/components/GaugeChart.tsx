import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

interface GaugeChartProps {
    value: number;
    max?: number;
    title: string;
    size?: number;
}

export const GaugeChart: React.FC<GaugeChartProps> = ({
    value,
    max = 5,
    title,
    size = 200,
}) => {
    const percentage = (value / max) * 100;

    // Determine color based on value
    const getColor = () => {
        if (value >= 4.0) return '#10b981'; // green
        if (value >= 3.5) return '#3b82f6'; // blue
        if (value >= 3.0) return '#f59e0b'; // amber
        return '#ef4444'; // red
    };

    const data = [
        { value: percentage },
        { value: 100 - percentage },
    ];

    return (
        <div className="flex flex-col items-center">
            <ResponsiveContainer width={size} height={size}>
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%"
                        cy="50%"
                        startAngle={180}
                        endAngle={0}
                        innerRadius="60%"
                        outerRadius="80%"
                        paddingAngle={0}
                        dataKey="value"
                    >
                        <Cell fill={getColor()} />
                        <Cell fill="#e5e7eb" />
                    </Pie>
                </PieChart>
            </ResponsiveContainer>
            <div className="text-center mt-[-30px]">
                <div className="text-4xl font-bold" style={{ color: getColor() }}>
                    {value.toFixed(1)}
                </div>
                <div className="text-sm text-gray-500">/ {max}</div>
                <div className="text-lg font-semibold mt-2 text-gray-700">{title}</div>
            </div>
        </div>
    );
};
