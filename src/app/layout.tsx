import Script from 'next/script';
import Nav from '../components/nav';
import { withBase } from '../lib/site';

// GA4 Measurement ID. Not secret (it ships in client-side code either way).
const GA_MEASUREMENT_ID = 'G-RXYRX9GG2T';

export const metadata = {
  title: 'Nitin | Portifolio',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const favicon = withBase('/icons/favicon/NClogo.svg');
  // Only load analytics in production builds so local `next dev` isn't tracked.
  const enableAnalytics = process.env.NODE_ENV === 'production';
  return (
    <html lang='en'>
      <head>
        <link rel='icon' type='image/svg+xml' href={favicon} />
        <link rel='alternate icon' href={favicon} />
        <link rel='apple-touch-icon' href={favicon} />
        <link rel='mask-icon' href={favicon} color='#000000' />
        <meta name='theme-color' content='#FFFFFF' />
      </head>
      <body className='relative bg-background font-body text-text'>
        <Nav />
        {children}
        {enableAnalytics && (
          <>
            <Script
              src={`https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`}
              strategy='afterInteractive'
            />
            <Script id='ga4-init' strategy='afterInteractive'>
              {`
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', '${GA_MEASUREMENT_ID}');
              `}
            </Script>
          </>
        )}
      </body>
    </html>
  );
}
