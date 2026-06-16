import React, { useState } from 'react';
import { Search, Brain, Lightbulb, ChevronDown, ChevronUp, Award } from 'lucide-react';

export const RubricPage: React.FC = () => {
    const [expandedSection, setExpandedSection] = useState<string>('curiosity');

    const toggleSection = (section: string) => {
        setExpandedSection(expandedSection === section ? '' : section);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 py-12">
            <div className="container mx-auto px-4 max-w-6xl">
                {/* Header */}
                <div className="text-center mb-12">
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary-100 text-primary-700 rounded-full mb-6 shadow-sm">
                        <Award className="w-4 h-4" />
                        <span className="text-sm font-semibold">Evaluation Framework</span>
                    </div>

                    <h1 className="text-5xl md:text-6xl font-extrabold mb-6">
                        <span className="bg-gradient-to-r from-primary-600 to-violet-600 bg-clip-text text-transparent">
                            C3 Soft Skills
                        </span>
                        <span className="block text-gray-900 mt-2">Rubric</span>
                    </h1>

                    <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
                        Standardized 5-point evaluation framework for assessing <strong>Curiosity, Critical Thinking, and Creativity</strong> in Software Engineering recruitment
                    </p>
                </div>

                {/* Curiosity Section */}
                <RubricSection
                    id="curiosity"
                    title="1. Curiosity"
                    definition="The desire to learn, understand, and explore new information or possibilities."
                    icon={<Search className="w-8 h-8" />}
                    iconBg="bg-gradient-to-br from-blue-500 to-cyan-500"
                    isExpanded={expandedSection === 'curiosity'}
                    onToggle={() => toggleSection('curiosity')}
                    big5Proxy="Openness"
                    exampleKeywords={['why', 'how does', 'curious', 'wonder', 'explore', 'find out', 'what if', 'investigate']}
                    scoreData={[
                        {
                            score: 1,
                            level: 'Passive / Indifferent',
                            description: 'Shows no interest in the topic; gives minimal, closed responses.',
                            example: '"I don\'t know.", "It is what it is."',
                            color: 'gray'
                        },
                        {
                            score: 2,
                            level: 'Low Interest',
                            description: 'Answers the question but seeks no further depth. Standard answers.',
                            example: '',
                            color: 'gray'
                        },
                        {
                            score: 3,
                            level: 'Moderate Curiosity',
                            description: 'Shows engagement; asks basic clarifying questions.',
                            example: '"How does that work?", "Can you explain more?"',
                            color: 'blue'
                        },
                        {
                            score: 4,
                            level: 'High Curiosity',
                            description: 'Actively seeks to understand the "why" and "how"; explores implications.',
                            example: '"Why did they choose that approach?", "What happens if we change X?"',
                            color: 'blue'
                        },
                        {
                            score: 5,
                            level: 'Deep / Epistemic',
                            description: 'Probes underlying mechanisms; challenges assumptions; seeks to bridge gaps.',
                            example: '"This connects to [Theory]... I wonder if [Hypothesis] would solve it?"',
                            color: 'blue',
                            isTop: true
                        }
                    ]}
                />

                {/* Critical Thinking Section */}
                <RubricSection
                    id="critical_thinking"
                    title="2. Critical Thinking"
                    definition="The objective analysis and evaluation of an issue in order to form a judgment."
                    icon={<Brain className="w-8 h-8" />}
                    iconBg="bg-gradient-to-br from-purple-500 to-pink-500"
                    isExpanded={expandedSection === 'critical_thinking'}
                    onToggle={() => toggleSection('critical_thinking')}
                    big5Proxy="Conscientiousness"
                    exampleKeywords={['because', 'therefore', 'however', 'evidence', 'root cause', 'evaluate', 'on the other hand', 'systematic']}
                    scoreData={[
                        {
                            score: 1,
                            level: 'Uncritical',
                            description: 'Accepts information blindly; relies on gut feeling or clichés without evidence.',
                            example: '"It just feels right.", "Everyone does it."',
                            color: 'gray'
                        },
                        {
                            score: 2,
                            level: 'Basic Understanding',
                            description: 'Can describe the issue but lacks analysis; misses obvious flaws.',
                            example: '',
                            color: 'gray'
                        },
                        {
                            score: 3,
                            level: 'Competent Analysis',
                            description: 'Identifies main pros/cons; follows a logical structure.',
                            example: '"On one hand X, on the other hand Y."',
                            color: 'purple'
                        },
                        {
                            score: 4,
                            level: 'Strong Evaluation',
                            description: 'Identifies biases, hidden assumptions, and edge cases; evaluates evidence quality.',
                            example: '"The argument assumes X, which might not be true in case Y."',
                            color: 'purple'
                        },
                        {
                            score: 5,
                            level: 'Advanced Synthesis',
                            description: 'Deconstructs complex systems; synthesizes multiple viewpoints; proposes solutions.',
                            example: '"The root cause isn\'t X, it\'s the interaction between Y and Z."',
                            color: 'purple',
                            isTop: true
                        }
                    ]}
                />

                {/* Creativity Section */}
                <RubricSection
                    id="creativity"
                    title="3. Creativity"
                    definition="The use of imagination or original ideas to produce novel solutions."
                    icon={<Lightbulb className="w-8 h-8" />}
                    iconBg="bg-gradient-to-br from-amber-500 to-orange-500"
                    isExpanded={expandedSection === 'creativity'}
                    onToggle={() => toggleSection('creativity')}
                    big5Proxy="Openness"
                    exampleKeywords={['what if', 'novel', 'alternative', 'prototype', 'reframe', 'unconventional', 'combined', 'brainstorm']}
                    scoreData={[
                        {
                            score: 1,
                            level: 'Conventional',
                            description: 'Sticks strictly to standard rules/procedures; no deviation.',
                            example: '"This is the standard way."',
                            color: 'gray'
                        },
                        {
                            score: 2,
                            level: 'Incremental',
                            description: 'Minor variations on existing ideas; safe choices.',
                            example: '"We could change the color."',
                            color: 'gray'
                        },
                        {
                            score: 3,
                            level: 'Adaptive',
                            description: 'Adapts existing solutions to new contexts; shows some flexibility.',
                            example: '"Like Uber, but for [X]."',
                            color: 'amber'
                        },
                        {
                            score: 4,
                            level: 'Original',
                            description: 'Proposes novel ideas; connects unrelated concepts (divergent thinking).',
                            example: '"What if we completely removed the interface?"',
                            color: 'amber'
                        },
                        {
                            score: 5,
                            level: 'Transformative',
                            description: 'Paradigm-shifting ideas; redefines the problem space; high novelty.',
                            example: '"Instead of solving X, we eliminate the need for X entirely."',
                            color: 'amber',
                            isTop: true
                        }
                    ]}
                />
            </div>
        </div>
    );
};

// Component Definitions
interface ScoreData {
    score: number;
    level: string;
    description: string;
    example: string;
    color: string;
    isTop?: boolean;
}

interface RubricSectionProps {
    id: string;
    title: string;
    definition: string;
    icon: React.ReactNode;
    iconBg: string;
    isExpanded: boolean;
    onToggle: () => void;
    scoreData: ScoreData[];
    big5Proxy: string;
    exampleKeywords: string[];
}

const RubricSection: React.FC<RubricSectionProps> = ({
    title,
    definition,
    icon,
    iconBg,
    isExpanded,
    onToggle,
    scoreData,
    big5Proxy,
    exampleKeywords
}) => (
    <section className="mb-8">
        <button
            onClick={onToggle}
            className="w-full card shadow-xl border-2 border-gray-100 hover:shadow-2xl transition-all p-6 text-left group"
        >
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4 flex-1">
                    <div className={`p-4 ${iconBg} rounded-xl text-white shadow-lg group-hover:scale-110 transition-transform`}>
                        {icon}
                    </div>
                    <div className="flex-1">
                        <h2 className="text-3xl font-bold text-gray-900 mb-2">{title}</h2>
                        <p className="text-gray-600 leading-relaxed mb-3">{definition}</p>
                        <div className="flex flex-wrap items-center gap-2">
                            <span className="px-2.5 py-1 rounded-full bg-violet-100 text-violet-700 text-xs font-semibold">
                                Big Five Proxy: {big5Proxy}
                            </span>
                            <span className="text-xs text-gray-400">Example language that boosts this score:</span>
                            {exampleKeywords.map(kw => (
                                <span key={kw} className="px-2.5 py-1 rounded-full bg-gray-100 text-gray-600 text-xs italic">
                                    "{kw}"
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
                <div className="ml-4">
                    {isExpanded ? (
                        <ChevronUp className="w-6 h-6 text-gray-400" />
                    ) : (
                        <ChevronDown className="w-6 h-6 text-gray-400" />
                    )}
                </div>
            </div>
        </button>

        {isExpanded && (
            <div className="mt-6 bg-white rounded-2xl shadow-xl border-2 border-gray-100 overflow-hidden animate-fadeIn">
                <table className="w-full">
                    <thead>
                        <tr className="bg-gradient-to-r from-gray-50 to-blue-50 border-b-2 border-gray-200">
                            <th className="px-6 py-4 font-bold text-gray-900 text-left w-24">Score</th>
                            <th className="px-6 py-4 font-bold text-gray-900 text-left w-48">Level</th>
                            <th className="px-6 py-4 font-bold text-gray-900 text-left">Description & Indicators</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                        {scoreData.map((item) => (
                            <ScoreRow key={item.score} data={item} />
                        ))}
                    </tbody>
                </table>
            </div>
        )}
    </section>
);

const ScoreRow: React.FC<{ data: ScoreData }> = ({ data }) => {
    const colorClasses = {
        gray: {
            bg: 'bg-white hover:bg-gray-50',
            badge: 'bg-gray-100 text-gray-700',
            text: 'text-gray-900',
            example: 'text-gray-500'
        },
        blue: {
            bg: data.isTop ? 'bg-blue-50 hover:bg-blue-100' : 'bg-white hover:bg-gray-50',
            badge: data.isTop ? 'bg-blue-500 text-white shadow-lg' : 'bg-blue-100 text-blue-700',
            text: data.isTop ? 'text-blue-900 font-bold' : 'text-gray-900',
            example: data.isTop ? 'text-blue-600' : 'text-gray-500'
        },
        purple: {
            bg: data.isTop ? 'bg-purple-50 hover:bg-purple-100' : 'bg-white hover:bg-gray-50',
            badge: data.isTop ? 'bg-purple-500 text-white shadow-lg' : 'bg-purple-100 text-purple-700',
            text: data.isTop ? 'text-purple-900 font-bold' : 'text-gray-900',
            example: data.isTop ? 'text-purple-600' : 'text-gray-500'
        },
        amber: {
            bg: data.isTop ? 'bg-amber-50 hover:bg-amber-100' : 'bg-white hover:bg-gray-50',
            badge: data.isTop ? 'bg-amber-500 text-white shadow-lg' : 'bg-amber-100 text-amber-700',
            text: data.isTop ? 'text-amber-900 font-bold' : 'text-gray-900',
            example: data.isTop ? 'text-amber-600' : 'text-gray-500'
        }
    };

    const colors = colorClasses[data.color as keyof typeof colorClasses] || colorClasses.gray;

    return (
        <tr className={`${colors.bg} transition-colors`}>
            <td className="px-6 py-5">
                <div className={`w-12 h-12 ${colors.badge} rounded-full flex items-center justify-center font-bold text-lg`}>
                    {data.score}
                </div>
            </td>
            <td className="px-6 py-5">
                <span className={`font-semibold ${colors.text} text-lg`}>{data.level}</span>
            </td>
            <td className="px-6 py-5">
                <p className="text-gray-700 leading-relaxed mb-2">{data.description}</p>
                {data.example && (
                    <p className={`text-sm ${colors.example} italic`}>
                        {data.example}
                    </p>
                )}
            </td>
        </tr>
    );
};
