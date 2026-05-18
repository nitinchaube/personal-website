/** @type {import('tailwindcss').Config} */
import plugin from 'tailwindcss/plugin';

module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    './content/**/*.{md,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        text: 'var(--color-text)',
        background: 'var(--color-background)',
        primary: 'var(--color-primary)',
        secondary: 'var(--color-secondary)',
        accent: 'var(--color-accent)',
      },
      boxShadow: {
        'timeline-border': `0 0 0 4px var(--color-accent)`,
        card: `0 0 5px 1px var(--color-text)`,
      },
      screens: {
        xs: '448px',
        'landscape-sm': { raw: '(max-width: 768px) and (orientation: landscape)' },
        'landscape-md': { raw: '(max-width: 1024px) and (orientation: landscape)' },
      },
      height: {
        screen: '100dvh',
        'screen-small': '100svh',
        'screen-large': '100lvh',
      },
      fontFamily: {
        // Used on the /notes section to evoke a quiet, paper-like reading experience.
        serif: [
          'Charter',
          'Iowan Old Style',
          'Source Serif Pro',
          'Apple Garamond',
          'Georgia',
          'Cambria',
          'Times New Roman',
          'serif',
        ],
      },
      typography: ({ theme }) => ({
        notes: {
          css: {
            '--tw-prose-body': '#222',
            '--tw-prose-headings': '#111',
            '--tw-prose-lead': '#3a3a3a',
            '--tw-prose-links': '#1a73e8',
            '--tw-prose-bold': '#111',
            '--tw-prose-counters': '#666',
            '--tw-prose-bullets': '#888',
            '--tw-prose-hr': '#e2e2e2',
            '--tw-prose-quotes': '#333',
            '--tw-prose-quote-borders': '#d4d4d4',
            '--tw-prose-captions': '#666',
            '--tw-prose-code': '#111',
            '--tw-prose-pre-code': '#e5e7eb',
            '--tw-prose-pre-bg': '#0d1117',
            '--tw-prose-th-borders': '#cbd5e1',
            '--tw-prose-td-borders': '#e5e7eb',
            fontFamily: 'Charter, "Iowan Old Style", "Source Serif Pro", "Apple Garamond", Georgia, serif',
            fontSize: '18px',
            lineHeight: '1.75',
            maxWidth: '720px',
            h1: {
              fontWeight: '700',
              fontSize: '2.1rem',
              lineHeight: '1.2',
              letterSpacing: '-0.01em',
            },
            h2: {
              fontWeight: '700',
              fontSize: '1.45rem',
              marginTop: '2.2em',
              marginBottom: '0.6em',
              letterSpacing: '-0.005em',
            },
            h3: {
              fontWeight: '600',
              fontStyle: 'italic',
              fontSize: '1.2rem',
              marginTop: '1.8em',
              marginBottom: '0.5em',
            },
            'h2 a, h3 a, h4 a': {
              textDecoration: 'none',
              color: 'inherit',
            },
            a: {
              textDecoration: 'underline',
              textUnderlineOffset: '3px',
              fontWeight: '500',
            },
            'a:hover': {
              color: '#0b57c4',
            },
            code: {
              fontFamily: 'SF-Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
              fontSize: '0.92em',
              background: '#f3f3f3',
              padding: '0.12em 0.35em',
              borderRadius: '4px',
              fontWeight: '500',
            },
            'code::before': { content: 'none' },
            'code::after': { content: 'none' },
            pre: {
              fontFamily: 'SF-Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
              fontSize: '14.5px',
              lineHeight: '1.6',
              borderRadius: '6px',
              padding: '1rem 1.1rem',
            },
            'pre code': {
              background: 'transparent',
              padding: '0',
              fontWeight: '400',
            },
            blockquote: {
              fontStyle: 'italic',
              borderLeftWidth: '3px',
            },
            img: {
              borderRadius: '4px',
              marginTop: '1.4em',
              marginBottom: '0.4em',
            },
            figure: {
              marginTop: '1.5em',
              marginBottom: '1.5em',
            },
            figcaption: {
              fontSize: '14px',
              color: '#666',
              textAlign: 'center',
              marginTop: '0.5em',
            },
            table: {
              fontSize: '15.5px',
            },
            hr: {
              marginTop: '3em',
              marginBottom: '3em',
            },
          },
        },
      }),
    },
    fontFamily: {
      body: ['SF-Pro', 'sans-serif'],
      mono: ['SF-Mono', 'monospace'],
    },
    backgroundSize: {
      big: '200%',
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
    plugin(function ({ matchUtilities, theme }) {
      matchUtilities(
        {
          'bg-gradient': (angle) => ({
            'background-image': `linear-gradient(${angle}, var(--tw-gradient-stops))`,
          }),
        },
        {
          values: Object.assign(theme('bgGradientDeg', {}), {
            10: '10deg',
            15: '15deg',
            20: '20deg',
            25: '25deg',
            30: '30deg',
            45: '45deg',
            60: '60deg',
            90: '90deg',
            120: '120deg',
            135: '135deg',
          }),
        }
      );
    }),
  ],
};
