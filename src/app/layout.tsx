import Nav from '../components/nav';

export const metadata = {
  title: 'Nitin | Portifolio',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang='en'>
      <head>
        <link rel='icon' type='image/svg+xml' href='/icons/favicon/NClogo.svg' />
        <link rel='alternate icon' href='/icons/favicon/NClogo.svg' />
        <link rel='apple-touch-icon' href='/icons/favicon/NClogo.svg' />
        <link rel='mask-icon' href='/icons/favicon/NClogo.svg' color='#000000' />
        <meta name='theme-color' content='#FFFFFF' />
      </head>
      <body className='relative bg-background font-body text-text'>
        <Nav />
        {children}
      </body>
    </html>
  );
}
