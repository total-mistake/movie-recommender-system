module.exports = {
    reactScriptsVersion: "react-scripts",
    style: {
        css: {
            loaderOptions: () => {
                return {
                    url: false,
                };
            },
        },
    },
    devServer: {
        proxy: {
            '/api': {
                target: 'http://localhost:8080',
                changeOrigin: true,
                secure: false,
                ws: true
            }
        }
    }
};