import Nav from '../components/nav';
import { withBase } from '../lib/site';

export const metadata = {
  title: 'Nitin | Portifolio',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const favicon = withBase('/icons/favicon/NClogo.svg');
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
      </body>
    </html>
  );
}
